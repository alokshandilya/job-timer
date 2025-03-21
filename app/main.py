from datetime import datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pytz import UTC, timezone  # For handling timezones
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Integer,
    Interval,
    create_engine,
    desc,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

# Database setup
DATABASE_URL = f"postgresql://{settings.database_username}:{settings.database_password}@{settings.database_hostname}:{settings.database_port}/{settings.database_name}?sslmode=require"
engine = create_engine(
    DATABASE_URL, pool_size=5, max_overflow=10, pool_timeout=30, pool_recycle=1800
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Timer(Base):
    __tablename__ = "office_timer"
    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime)
    pause_time = Column(DateTime)
    resume_time = Column(DateTime)
    total_elapsed = Column(Interval, default=timedelta())
    is_running = Column(Boolean, default=False)
    date = Column(Date)  # To track the date
    first_in_time = Column(DateTime)  # First IN time of the day
    last_out_time = Column(DateTime)  # Last OUT time of the day


Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# IST timezone
IST = timezone("Asia/Kolkata")


# Dependency to manage database sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Helper function to get current time in IST as timezone-aware datetime
def get_current_time():
    # This returns a timezone-aware datetime in IST
    return datetime.now(IST)


# Helper function to get the current date in IST
def get_current_date():
    return get_current_time().date()


# Helper function to normalize datetime for comparison
# This ensures all datetimes use the same timezone for math operations
def normalize_datetime(dt):
    if dt is None:
        return None

    # If datetime is naive, assume it's in UTC and make it aware
    if dt.tzinfo is None:
        dt = UTC.localize(dt)

    # Convert to IST for consistency
    return dt.astimezone(IST)


# Helper function to format datetime for display
def format_time(dt):
    if dt is None:
        return "N/A"
    # Normalize to ensure consistent timezone
    dt = normalize_datetime(dt)
    return dt.strftime("%I:%M %p")  # 12-hour format with AM/PM


# Helper function to format date for display
def format_date(dt):
    if dt is None:
        return "N/A"
    return dt.strftime("%d-%b-%Y")  # Format: 21-Mar-2023


# Helper function to get day of week
def get_day_of_week(dt):
    if dt is None:
        return "N/A"
    return dt.strftime("%A")  # Full day name


# Helper function to format timedelta for display
def format_timedelta(td):
    if td is None:
        return "00:00:00"

    total_seconds = td.total_seconds()
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def get_timer_state(db: Session):
    # Get current date in IST
    current_date_ist = get_current_date()

    try:
        # Try to get today's timer first
        timer = db.query(Timer).filter(Timer.date == current_date_ist).first()

        # If no timer for today exists, get the most recent timer
        if not timer:
            timer = db.query(Timer).order_by(Timer.date.desc()).first()

            # If there's a previous timer but it's not from today, we need a new one
            if timer and timer.date != current_date_ist:
                timer = Timer(
                    is_running=False,
                    total_elapsed=timedelta(),
                    date=current_date_ist,
                    start_time=None,
                    pause_time=None,
                    resume_time=None,
                )
                db.add(timer)
                db.commit()
                db.refresh(timer)
            # If no timer exists at all, create a new one
            elif not timer:
                timer = Timer(
                    is_running=False,
                    total_elapsed=timedelta(),
                    date=current_date_ist,
                    start_time=None,
                    pause_time=None,
                    resume_time=None,
                )
                db.add(timer)
                db.commit()
                db.refresh(timer)

        # Normalize datetime fields for consistent timezone handling
        timer.start_time = normalize_datetime(timer.start_time)
        timer.pause_time = normalize_datetime(timer.pause_time)
        timer.resume_time = normalize_datetime(timer.resume_time)

        # Handle case where first_in_time and last_out_time columns might not exist yet
        try:
            timer.first_in_time = normalize_datetime(timer.first_in_time)
            timer.last_out_time = normalize_datetime(timer.last_out_time)
        except AttributeError:
            # If the columns don't exist yet, we'll handle that in the template
            pass

        return timer

    except Exception as e:
        # Log the error and return a minimal timer object to prevent crashing
        print(f"Error in get_timer_state: {e}")
        return Timer(
            is_running=False,
            total_elapsed=timedelta(),
            date=current_date_ist,
            start_time=None,
            pause_time=None,
            resume_time=None,
        )


# Helper function to get history records for the past 7 days
def get_history_records(db: Session):
    current_date = get_current_date()
    start_date = current_date - timedelta(days=6)  # Last 7 days including today

    records = (
        db.query(Timer)
        .filter(Timer.date >= start_date)
        .order_by(desc(Timer.date))
        .all()
    )

    formatted_records = []
    for record in records:
        # Calculate total hours for completed days
        if record.date == current_date:
            # For today, use the current elapsed time
            timer = get_timer_state(db)
            total_elapsed = timer.total_elapsed

            # If timer is running, add current session time
            if timer.is_running and timer.start_time:
                current_time = get_current_time()
                current_elapsed = current_time - timer.start_time
                total_elapsed += current_elapsed
        else:
            # For past days, use the recorded total_elapsed
            total_elapsed = record.total_elapsed

        formatted_records.append(
            {
                "date": format_date(record.date),
                "day": get_day_of_week(record.date),
                "first_in": format_time(record.first_in_time)
                if record.first_in_time
                else "N/A",
                "last_out": format_time(record.last_out_time)
                if record.last_out_time
                else "N/A",
                "total_hours": format_timedelta(total_elapsed),
            }
        )

    # Pad with empty days if we have fewer than 7 days of records
    while len(formatted_records) < 7:
        formatted_records.append(
            {
                "date": "N/A",
                "day": "N/A",
                "first_in": "N/A",
                "last_out": "N/A",
                "total_hours": "00:00:00",
            }
        )

    return formatted_records


@app.get("/current-datetime")
async def get_current_datetime():
    now = get_current_time()
    return now.strftime("%A, %d %B %Y - %I:%M:%S %p")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    timer = get_timer_state(db)
    current_time = get_current_time()

    # Format elapsed time to be more readable (HH:MM:SS)
    total_seconds = timer.total_elapsed.total_seconds()
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed_time = f"{hours:02}:{minutes:02}:{seconds:02}"

    # Determine if it's a new day or first visit
    current_date_ist = get_current_date()
    is_new_day = timer.date == current_date_ist and timer.start_time is None

    # Calculate current elapsed time if timer is running
    if timer.is_running and timer.start_time:
        current_elapsed = current_time - timer.start_time
        total_seconds = (
            timer.total_elapsed.total_seconds() + current_elapsed.total_seconds()
        )
        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_time = f"{hours:02}:{minutes:02}:{seconds:02}"

    # Get history records - safely handle in case of column missing errors
    try:
        history_records = get_history_records(db)
    except Exception as e:
        print(f"Error getting history records: {e}")
        history_records = []

    # Format current datetime for display
    current_datetime = current_time.strftime("%A, %d %B %Y - %I:%M:%S %p")

    # Safely get first_in_time and last_activity_time with fallbacks
    try:
        first_in_time = format_time(getattr(timer, "first_in_time", None))
    except:
        first_in_time = "N/A"

    try:
        last_out = getattr(timer, "last_out_time", None)
        last_activity_time = format_time(last_out if last_out else timer.start_time)
    except:
        last_activity_time = format_time(timer.start_time)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "is_running": timer.is_running,
            "elapsed_time": elapsed_time,
            "is_new_day": is_new_day,
            "history_records": history_records,
            "current_datetime": current_datetime,
            "first_in_time": first_in_time,
            "last_activity_time": last_activity_time,
        },
    )


@app.post("/start", response_class=HTMLResponse)
async def start_timer(request: Request, db: Session = Depends(get_db)):
    timer = get_timer_state(db)
    current_date_ist = get_current_date()

    # Check if the timer is already running
    if timer.is_running:
        raise HTTPException(status_code=400, detail="Timer is already running.")

    # Only allow starting if it's a new day or first start of the day
    if timer.date == current_date_ist and timer.start_time is None:
        # Store the current time
        now = get_current_time()
        timer.start_time = now
        timer.is_running = True

        # Record first IN time of the day
        timer.first_in_time = now

        db.commit()
        db.refresh(timer)
        # Normalize after DB refresh
        timer.start_time = normalize_datetime(timer.start_time)
        timer.first_in_time = normalize_datetime(timer.first_in_time)
    else:
        raise HTTPException(status_code=400, detail="Timer has already started today.")

    # Return the updated page
    return await read_root(request, db)


@app.post("/pause", response_class=HTMLResponse)
async def pause_timer(request: Request, db: Session = Depends(get_db)):
    timer = get_timer_state(db)

    # Check if the timer is running
    if not timer.is_running:
        raise HTTPException(status_code=400, detail="Timer is not running.")

    now = get_current_time()

    # Calculate elapsed time since last start/resume
    if timer.start_time:
        # Both datetimes are normalized to IST
        elapsed = now - timer.start_time
        timer.total_elapsed += elapsed

    timer.pause_time = now
    timer.is_running = False

    # Update last OUT time
    timer.last_out_time = now

    db.commit()
    db.refresh(timer)

    # Normalize after DB refresh
    timer.pause_time = normalize_datetime(timer.pause_time)
    timer.start_time = normalize_datetime(timer.start_time)
    timer.last_out_time = normalize_datetime(timer.last_out_time)

    # Return the updated page
    return await read_root(request, db)


@app.post("/resume", response_class=HTMLResponse)
async def resume_timer(request: Request, db: Session = Depends(get_db)):
    timer = get_timer_state(db)

    # Check if the timer is already running
    if timer.is_running:
        raise HTTPException(status_code=400, detail="Timer is already running.")

    # Check if the timer has been started and then paused
    if timer.start_time is None:
        raise HTTPException(status_code=400, detail="Timer has not been started yet.")

    if timer.pause_time is None:
        raise HTTPException(status_code=400, detail="Timer has not been paused yet.")

    # Set resume time and update running status
    now = get_current_time()
    timer.resume_time = now
    timer.start_time = now  # Reset start_time for current session
    timer.pause_time = None  # Reset pause_time
    timer.is_running = True
    db.commit()
    db.refresh(timer)

    # Normalize after DB refresh
    timer.start_time = normalize_datetime(timer.start_time)
    timer.resume_time = normalize_datetime(timer.resume_time)

    # Return the updated page
    return await read_root(request, db)


@app.get("/status")
async def get_status(db: Session = Depends(get_db)):
    timer = get_timer_state(db)

    # Calculate current elapsed time
    total_seconds = timer.total_elapsed.total_seconds()

    # If timer is running, add the current session time
    if timer.is_running and timer.start_time:
        current_time = get_current_time()
        # Both datetimes are normalized to IST
        current_elapsed = current_time - timer.start_time
        total_seconds += current_elapsed.total_seconds()

    # Format the elapsed time
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed_time = f"{hours:02}:{minutes:02}:{seconds:02}"

    return elapsed_time  # Just return the formatted time string
