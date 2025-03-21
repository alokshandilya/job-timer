from datetime import datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pytz import UTC, timezone  # For handling timezones
from sqlalchemy import Boolean, Column, Date, DateTime, Integer, Interval, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

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
    date = Column(Date)  # New column to track the date


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


# Helper function to get the current timer state
def get_timer_state(db: Session):
    # Get current date in IST
    current_date_ist = get_current_date()

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

    return timer


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    timer = get_timer_state(db)

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
        current_time = get_current_time()
        # Both datetimes are now normalized to IST
        current_elapsed = current_time - timer.start_time
        total_seconds = (
            timer.total_elapsed.total_seconds() + current_elapsed.total_seconds()
        )
        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_time = f"{hours:02}:{minutes:02}:{seconds:02}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "is_running": timer.is_running,
            "elapsed_time": elapsed_time,
            "is_new_day": is_new_day,
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
        db.commit()
        db.refresh(timer)
        # Normalize after DB refresh
        timer.start_time = normalize_datetime(timer.start_time)
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
    db.commit()
    db.refresh(timer)

    # Normalize after DB refresh
    timer.pause_time = normalize_datetime(timer.pause_time)
    timer.start_time = normalize_datetime(timer.start_time)

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
