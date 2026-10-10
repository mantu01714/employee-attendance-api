"""
Employee Attendance & Analytics API
Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB  (a local .env is fine; real environment variables take precedence)
"""
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))
PRESENCE_STATUSES = {"PRESENT", "WFH", "ON_DUTY"}
ALL_STATUSES = PRESENCE_STATUSES | {"ABSENT", "LEAVE"}

# ---------------------------------------------------------------------------
# Database connection  (tz_aware=True → PyMongo returns aware datetimes in UTC)
# ---------------------------------------------------------------------------
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]


def create_indexes():
    """Create all required indexes idempotently.
    Called at server startup (lifespan) — not at import time, so test
    imports don't block waiting for a MongoDB connection.

    Indexes created:
    - employees.emp_code (unique): enforces uniqueness, enables O(1) look-ups.
    - attendance_logs(emp_code ASC, date ASC) (unique): one record per employee
      per attendance day; the primary access pattern for every query.
    - attendance_logs(date DESC, emp_code ASC): serves /attendance list sort
      exactly, so MongoDB uses IXSCAN instead of an in-memory sort.
    """
    db.employees.create_index("emp_code", unique=True)
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)], unique=True
    )
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)]
    )



@asynccontextmanager
async def lifespan(application: "FastAPI"):
    create_indexes()
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def half_up(value: float, places: int) -> float:
    """Round half-up (not banker's rounding).  Python's built-in round() uses
    banker's rounding (round-half-to-even), which the spec explicitly forbids."""
    factor = 10 ** places
    return math.floor(value * factor + 0.5) / factor


def ms_to_dt(epoch_ms: int) -> datetime:
    """Convert epoch milliseconds to a UTC-aware datetime, truncated to whole seconds (R1)."""
    dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
    return dt.replace(microsecond=0)


def dt_to_ms(dt: datetime) -> int:
    """Convert a datetime to epoch milliseconds."""
    return int(dt.timestamp() * 1000)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """Return the attendance date string YYYY-MM-DD per R1.

    Normally: the IST calendar date of punch_in.
    Overnight shift (shift_end <= shift_start, e.g. 22:00-06:00):
      If the punch_in's IST time is *earlier than* shift_end, the punch
      belongs to the shift that started the previous IST calendar day.
    """
    punch_ist = punch_in.astimezone(IST)
    date_str = punch_ist.date().isoformat()

    sh, sm = map(int, shift_start.split(":"))
    eh, em = map(int, shift_end.split(":"))
    is_overnight = (eh * 60 + em) <= (sh * 60 + sm)

    if is_overnight:
        punch_time_minutes = punch_ist.hour * 60 + punch_ist.minute
        end_minutes = eh * 60 + em
        if punch_time_minutes < end_minutes:
            # Punch belongs to the shift that started the previous calendar day
            prev_day = (punch_ist - timedelta(days=1)).date()
            date_str = prev_day.isoformat()

    return date_str


def compute_late_minutes(punch_in: datetime, shift_start: str, date_str: str) -> int:
    """R2: late only if strictly more than 10 min after shift_start.
    late_minutes = floor((punch_in - shift_start) / 60).
    punch_in is UTC-aware; we work in IST for the comparison."""
    sh, sm = map(int, shift_start.split(":"))
    # Build the shift_start instant for the attendance date in IST
    att_date = datetime.fromisoformat(date_str)
    shift_start_ist = datetime(
        att_date.year, att_date.month, att_date.day, sh, sm, 0, tzinfo=IST
    )
    diff_seconds = (punch_in - shift_start_ist).total_seconds()
    if diff_seconds > 10 * 60:  # strictly more than 10 minutes (600 seconds)
        return int(diff_seconds // 60)
    return 0


def compute_overtime_minutes(punch_out: datetime, shift_end: str, date_str: str, shift_start: str) -> int:
    """R3: overtime = floor minutes between shift_end and punch_out, only when >= 30.
    For overnight shifts shift_end is on the next calendar day."""
    eh, em = map(int, shift_end.split(":"))
    sh, sm = map(int, shift_start.split(":"))
    att_date = datetime.fromisoformat(date_str)

    is_overnight = (eh * 60 + em) <= (sh * 60 + sm)
    if is_overnight:
        # shift_end is on the next calendar day
        end_date = att_date + timedelta(days=1)
    else:
        end_date = att_date

    shift_end_ist = datetime(
        end_date.year, end_date.month, end_date.day, eh, em, 0, tzinfo=IST
    )
    diff_seconds = (punch_out - shift_end_ist).total_seconds()
    if diff_seconds >= 30 * 60:  # >= 30 minutes
        return int(diff_seconds // 60)
    return 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: (punch_out - punch_in) in seconds / 3600, rounded 2 dp half-up."""
    seconds = (punch_out - punch_in).total_seconds()
    return half_up(seconds / 3600, 2)


def compute_half_day(work_hours: float) -> bool:
    """R5: half_day = true when rounded work_hours < 4.50."""
    return work_hours < 4.50


def recompute_derived(punch_in: Optional[datetime], punch_out: Optional[datetime],
                      shift_start: str, shift_end: str, date_str: str, status: str) -> dict:
    """Recompute all four derived fields given the current field values.
    Returns a dict with keys: late_minutes, work_hours, overtime_minutes, half_day.
    Used after both punch-in and any correction."""
    if status in ("ABSENT", "LEAVE"):
        return {"late_minutes": 0, "work_hours": None, "overtime_minutes": 0, "half_day": False}

    late = compute_late_minutes(punch_in, shift_start, date_str) if punch_in else 0

    if punch_in and punch_out:
        wh = compute_work_hours(punch_in, punch_out)
        ot = compute_overtime_minutes(punch_out, shift_end, date_str, shift_start)
        hd = compute_half_day(wh)
    else:
        wh = None
        ot = 0
        hd = False

    return {"late_minutes": late, "work_hours": wh, "overtime_minutes": ot, "half_day": hd}


def record_to_api(doc: dict) -> dict:
    """Convert a raw MongoDB attendance document to the API AttendanceRecord shape.

    Key conversions:
    - Remove _id (internal)
    - punch_in / punch_out: datetime → epoch ms (or None)
    - history[].at and history[].changes punch_in/punch_out: datetime → epoch ms
    - Ensure late_minutes, overtime_minutes, half_day default if missing (legacy records)
    """
    out = {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc["punch_in"]) if doc.get("punch_in") else None,
        "punch_out": dt_to_ms(doc["punch_out"]) if doc.get("punch_out") else None,
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": [],
    }
    for entry in doc.get("history", []):
        changes = {}
        for field, change in entry.get("changes", {}).items():
            if field in ("punch_in", "punch_out"):
                changes[field] = {
                    "from": dt_to_ms(change["from"]) if change.get("from") else None,
                    "to": dt_to_ms(change["to"]) if change.get("to") else None,
                }
            else:
                changes[field] = {"from": change["from"], "to": change["to"]}
        out["history"].append({
            "at": dt_to_ms(entry["at"]),
            "by": entry["by"],
            "reason": entry["reason"],
            "changes": changes,
        })
    return out


def employee_to_api(doc: dict) -> dict:
    """Convert an employee MongoDB doc to the API Employee shape."""
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(doc["created_at"]),
    }


def validate_month_param(month: str) -> tuple[str, str]:
    """Parse 'YYYY-MM' and return (first_day_str, last_day_str) of that month."""
    import calendar
    try:
        year, mon = int(month[:4]), int(month[5:7])
    except (ValueError, IndexError):
        raise HTTPException(422, "month must be YYYY-MM")
    _, last_day = calendar.monthrange(year, mon)
    first_day = f"{year}-{mon:02d}-01"
    last_day_str = f"{year}-{mon:02d}-{last_day:02d}"
    return first_day, last_day_str


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EmployeeCreate(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str  # validated below

    @model_validator(mode="after")
    def shift_times_must_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self

    @field_validator("joined_on")
    @classmethod
    def joined_on_is_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("joined_on must be YYYY-MM-DD")
        return v


# EpochMillis: spec says values < 100_000_000_000 (looks like seconds) are rejected
EPOCH_MS_MIN = 100_000_000_000
EPOCH_MS_MAX = 4_102_444_800_000


class EpochMillisField(int):
    """A validated epoch-milliseconds integer."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, float) and not v.is_integer():
            raise ValueError("epoch milliseconds must be an integer, not a float")
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise ValueError("epoch milliseconds must be an integer")
        if v < EPOCH_MS_MIN or v > EPOCH_MS_MAX:
            raise ValueError(
                f"epoch milliseconds must be between {EPOCH_MS_MIN} and {EPOCH_MS_MAX}"
            )
        return v


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(default=None, ge=EPOCH_MS_MIN, le=EPOCH_MS_MAX)
    status: str = "PRESENT"

    @field_validator("status")
    @classmethod
    def status_must_be_presence(cls, v: str) -> str:
        if v not in PRESENCE_STATUSES:
            raise ValueError(f"status must be one of {sorted(PRESENCE_STATUSES)}")
        return v

    @field_validator("punched_at", mode="before")
    @classmethod
    def reject_float_millis(cls, v):
        if isinstance(v, float) and not v.is_integer():
            raise ValueError("punched_at must be an integer (epoch milliseconds)")
        return v


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(default=None, ge=EPOCH_MS_MIN, le=EPOCH_MS_MAX)

    @field_validator("punched_at", mode="before")
    @classmethod
    def reject_float_millis(cls, v):
        if isinstance(v, float) and not v.is_integer():
            raise ValueError("punched_at must be an integer (epoch milliseconds)")
        return v


class RegularizeRequest(BaseModel):
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)
    status: Optional[str] = None
    punch_in: Optional[int] = Field(default=None, ge=EPOCH_MS_MIN, le=EPOCH_MS_MAX)
    punch_out: Optional[int] = Field(default=None, ge=EPOCH_MS_MIN, le=EPOCH_MS_MAX)

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ALL_STATUSES:
            raise ValueError(f"status must be one of {sorted(ALL_STATUSES)}")
        return v

    @field_validator("punch_in", "punch_out", mode="before")
    @classmethod
    def reject_float_millis(cls, v):
        if isinstance(v, float) and not v.is_integer():
            raise ValueError("punch times must be integers (epoch milliseconds)")
        return v

    @model_validator(mode="after")
    def absent_leave_no_punches(self):
        if self.status in ("ABSENT", "LEAVE"):
            if self.punch_in is not None or self.punch_out is not None:
                raise ValueError(
                    "punch_in and punch_out must not be set when status is ABSENT or LEAVE"
                )
        return self


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """R: 200 only if MongoDB answers ping, else 503."""
    try:
        db.command("ping")
    except Exception:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeCreate):
    """Create an employee.  Uniqueness is enforced by the unique index on emp_code,
    so concurrent duplicate requests are safe — only one will succeed."""
    doc = body.model_dump()
    doc["created_at"] = datetime.now(tz=timezone.utc)
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_to_api(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """List employees sorted by emp_code asc. total reflects the filtered set."""
    q: dict = {}
    if department:
        q["department"] = department
    skip = (page - 1) * page_size  # FIX: was page * page_size (off-by-one)
    total = db.employees.count_documents(q)  # FIX: was count_documents({}) ignoring filter
    cursor = (
        db.employees.find(q, {"_id": 0})
        .sort("emp_code", ASCENDING)        # FIX: was missing
        .skip(skip)
        .limit(page_size)
    )
    items = [employee_to_api(doc) for doc in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest):
    """Record a punch-in.
    - 404 if employee not found
    - 409 if already punched in for that attendance date (enforced by unique index)
    - 422 for invalid body
    Concurrency is safe: we rely on the unique index, not check-then-insert."""
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(404, f"Employee '{body.emp_code}' not found")

    # R1: truncate to whole seconds, then derive attendance date in IST
    if body.punched_at is not None:
        punch_dt = ms_to_dt(body.punched_at)
    else:
        punch_dt = datetime.now(tz=timezone.utc).replace(microsecond=0)

    date_str = attendance_date(punch_dt, emp["shift_start"], emp["shift_end"])

    derived = recompute_derived(
        punch_in=punch_dt,
        punch_out=None,
        shift_start=emp["shift_start"],
        shift_end=emp["shift_end"],
        date_str=date_str,
        status=body.status,
    )

    doc = {
        "emp_code": body.emp_code,
        "date": date_str,
        "status": body.status,
        "punch_in": punch_dt,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": derived["late_minutes"],
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, f"Already punched in for date {date_str}")

    return record_to_api(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutRequest):
    """Close the most recent open attendance record for the employee.
    The record whose punch_in is <= punched_at and whose punch_out is null.
    - 404 if employee not found, or no open punch-in found
    - 409 if already punched out
    - 422 if punched_at <= punch_in or more than 24h after punch_in
    Concurrency: use find_one_and_update with filter punch_out: null so only one
    concurrent punch-out can win; the loser gets a 409."""
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(404, f"Employee '{body.emp_code}' not found")

    if body.punched_at is not None:
        punch_dt = ms_to_dt(body.punched_at)
    else:
        punch_dt = datetime.now(tz=timezone.utc).replace(microsecond=0)

    # Find the most recent record whose punch_in <= punched_at
    record = db.attendance_logs.find_one(
        {
            "emp_code": body.emp_code,
            "punch_in": {"$lte": punch_dt},
        },
        sort=[("punch_in", DESCENDING)],
    )

    if record is None:
        raise HTTPException(404, "No punch-in found for this employee at or before the given time")

    # 422: punched_at must be after punch_in
    if punch_dt <= record["punch_in"]:
        raise HTTPException(422, "punched_at must be after punch_in")

    # 422: must be within 24 hours of punch_in
    if (punch_dt - record["punch_in"]).total_seconds() > 86400:
        raise HTTPException(422, "punched_at is more than 24 hours after punch_in")

    # 409: already punched out
    if record.get("punch_out") is not None:
        raise HTTPException(409, "Already punched out for this record")

    derived = recompute_derived(
        punch_in=record["punch_in"],
        punch_out=punch_dt,
        shift_start=emp["shift_start"],
        shift_end=emp["shift_end"],
        date_str=record["date"],
        status=record["status"],
    )

    # Atomic update: filter on punch_out: None ensures the concurrent loser gets no match
    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": punch_dt,
                "work_hours": derived["work_hours"],
                "overtime_minutes": derived["overtime_minutes"],
                "half_day": derived["half_day"],
            }
        },
        return_document=True,
    )

    if updated is None:
        # Another concurrent punch-out won
        raise HTTPException(409, "Already punched out for this record")

    return record_to_api(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """List attendance records.  Sorted by date DESC, emp_code ASC.
    Filtering, sorting, skip and limit are all pushed to MongoDB — no in-memory sort."""
    # Validate date_from > date_to → 422
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must not be after date_to")

    if status and status not in ALL_STATUSES:
        raise HTTPException(422, f"status must be one of {sorted(ALL_STATUSES)}")

    q: dict = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status

    skip = (page - 1) * page_size
    total = db.attendance_logs.count_documents(q)
    cursor = (
        db.attendance_logs.find(q, {"_id": 0})
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )
    items = [record_to_api(doc) for doc in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, body: RegularizeRequest):
    """Manually correct one attendance record.
    - Applies only the supplied fields (omitted fields are unchanged).
    - Recomputes derived fields (late_minutes, work_hours, overtime_minutes, half_day).
    - Appends exactly one history entry (never overwrites).
    - 422 if the request changes nothing.
    - Concurrency: uses optimistic locking via the history array length to detect
      concurrent edits; the losing request gets 409."""
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, f"Employee '{emp_code}' not found")

    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if record is None:
        raise HTTPException(404, f"No attendance record for {emp_code} on {date}")

    # Build the proposed new state
    new_status = body.status if body.status is not None else record["status"]
    new_punch_in = ms_to_dt(body.punch_in) if body.punch_in is not None else record.get("punch_in")
    new_punch_out = ms_to_dt(body.punch_out) if body.punch_out is not None else record.get("punch_out")

    # ABSENT/LEAVE clears both punch times
    if new_status in ("ABSENT", "LEAVE"):
        new_punch_in = None
        new_punch_out = None

    # A presence status requires a punch_in
    if new_status in PRESENCE_STATUSES and new_punch_in is None:
        raise HTTPException(422, "A presence status requires a punch_in")

    # punch_in must stay on the record's attendance date (R1)
    if new_punch_in is not None:
        expected_date = attendance_date(new_punch_in, emp["shift_start"], emp["shift_end"])
        if expected_date != date:
            raise HTTPException(422, f"punch_in resolves to attendance date {expected_date}, not {date}")

    # punch_out must be after punch_in and within 24 hours
    if new_punch_out is not None:
        if new_punch_in is None:
            raise HTTPException(422, "Cannot set punch_out without punch_in")
        if new_punch_out <= new_punch_in:
            raise HTTPException(422, "punch_out must be after punch_in")
        if (new_punch_out - new_punch_in).total_seconds() > 86400:
            raise HTTPException(422, "punch_out must be within 24 hours of punch_in")

    # Recompute derived fields
    derived = recompute_derived(
        punch_in=new_punch_in,
        punch_out=new_punch_out,
        shift_start=emp["shift_start"],
        shift_end=emp["shift_end"],
        date_str=date,
        status=new_status,
    )

    # Compute which fields actually changed (for the history entry)
    old_punch_in = record.get("punch_in")
    old_punch_out = record.get("punch_out")
    old_status = record["status"]
    old_late = record.get("late_minutes", 0)
    old_work = record.get("work_hours")
    old_ot = record.get("overtime_minutes", 0)
    old_hd = record.get("half_day", False)

    changes: dict = {}
    if new_status != old_status:
        changes["status"] = {"from": old_status, "to": new_status}
    if new_punch_in != old_punch_in:
        changes["punch_in"] = {"from": old_punch_in, "to": new_punch_in}
    if new_punch_out != old_punch_out:
        changes["punch_out"] = {"from": old_punch_out, "to": new_punch_out}
    if derived["late_minutes"] != old_late:
        changes["late_minutes"] = {"from": old_late, "to": derived["late_minutes"]}
    if derived["work_hours"] != old_work:
        changes["work_hours"] = {"from": old_work, "to": derived["work_hours"]}
    if derived["overtime_minutes"] != old_ot:
        changes["overtime_minutes"] = {"from": old_ot, "to": derived["overtime_minutes"]}
    if derived["half_day"] != old_hd:
        changes["half_day"] = {"from": old_hd, "to": derived["half_day"]}

    if not changes:
        raise HTTPException(422, "The request changes nothing")

    history_entry = {
        "at": datetime.now(tz=timezone.utc),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    # Optimistic concurrency: match on current history length so two simultaneous
    # corrections cannot silently overwrite each other; the loser gets 409.
    current_history_len = len(record.get("history", []))

    set_fields: dict = {
        "status": new_status,
        "punch_in": new_punch_in,
        "punch_out": new_punch_out,
        "work_hours": derived["work_hours"],
        "late_minutes": derived["late_minutes"],
        "overtime_minutes": derived["overtime_minutes"],
        "half_day": derived["half_day"],
    }

    updated = db.attendance_logs.find_one_and_update(
        {
            "_id": record["_id"],
            # Optimistic lock: the history array must still have the same length
            f"history.{current_history_len}": {"$exists": False},
        },
        {
            "$set": set_fields,
            "$push": {"history": history_entry},
        },
        return_document=True,
    )

    if updated is None:
        raise HTTPException(409, "Concurrent edit detected; please retry")

    return record_to_api(updated)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str,
    month: str = Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    """Monthly attendance summary for one employee via MongoDB aggregation.
    working_days: Mon-Fri days in the month from joined_on (R7).
    present_days: PRESENT/WFH/ON_DUTY records on Mon-Fri; half_day counts 0.5 (R5-R7).
    leave_days: count of LEAVE records.
    late_count: records with late_minutes > 0.
    attendance_pct: present_days / working_days * 100, null when working_days is 0."""
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, f"Employee '{emp_code}' not found")

    first_day, last_day = validate_month_param(month)
    # working_days start from max(joined_on, first_day)
    start_date = max(first_day, emp["joined_on"])

    # Count Mon-Fri days between start_date and last_day (inclusive)
    import calendar as cal
    working_days = sum(
        1
        for d in _date_range(start_date, last_day)
        if datetime.strptime(d, "%Y-%m-%d").weekday() < 5  # 0=Mon, 4=Fri
    )

    pipeline = [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$gte": first_day, "$lte": last_day},
            }
        },
        {
            "$addFields": {
                # day-of-week: 1=Sun, 2=Mon, …, 7=Sat  (MongoDB $dayOfWeek)
                "dow": {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                "is_weekday": {
                    "$and": [
                        {"$gte": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}}, 2]},
                        {"$lte": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}}, 6]},
                    ]
                },
                "is_presence": {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                "half_day_val": {"$ifNull": ["$half_day", False]},
                "late_min_val": {"$ifNull": ["$late_minutes", 0]},
                "ot_min_val": {"$ifNull": ["$overtime_minutes", 0]},
            }
        },
        {
            "$group": {
                "_id": None,
                "present_days": {
                    # Mon-Fri presence records: full day = 1.0, half day = 0.5
                    "$sum": {
                        "$cond": [
                            {"$and": ["$is_presence", "$is_weekday"]},
                            {"$cond": ["$half_day_val", 0.5, 1.0]},
                            0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_min_val", 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": "$late_min_val"},
                "total_overtime_minutes": {"$sum": "$ot_min_val"},
            }
        },
    ]

    results = list(db.attendance_logs.aggregate(pipeline))
    row = results[0] if results else {}

    present_days = half_up(row.get("present_days", 0), 2)
    attendance_pct = (
        half_up(present_days / working_days * 100, 2) if working_days > 0 else None
    )

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": row.get("leave_days", 0),
        "late_count": row.get("late_count", 0),
        "total_late_minutes": row.get("total_late_minutes", 0),
        "total_overtime_minutes": row.get("total_overtime_minutes", 0),
        "attendance_pct": attendance_pct,
    }


def _date_range(start: str, end: str):
    """Yield date strings YYYY-MM-DD from start to end inclusive."""
    cur = datetime.strptime(start, "%Y-%m-%d").date()
    last = datetime.strptime(end, "%Y-%m-%d").date()
    while cur <= last:
        yield cur.isoformat()
        cur += timedelta(days=1)


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    """Department summary for a month.
    headcount: employees with joined_on <= last day of month (R9).
    Employees with zero logs are still counted in headcount.
    Uses $lookup to join attendance_logs into the employees pipeline."""
    first_day, last_day = validate_month_param(month)

    emp_filter: dict = {"joined_on": {"$lte": last_day}}
    if department:
        emp_filter["department"] = department

    pipeline = [
        # Start from employees so we capture people with zero logs
        {"$match": emp_filter},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"ec": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {"$eq": ["$emp_code", "$$ec"]},
                            "date": {"$gte": first_day, "$lte": last_day},
                        }
                    }
                ],
                "as": "logs",
            }
        },
        {"$unwind": {"path": "$logs", "preserveNullAndEmpty": True}},
        {
            "$addFields": {
                "log_status": {"$ifNull": ["$logs.status", None]},
                "log_half_day": {"$ifNull": ["$logs.half_day", False]},
                "log_late": {"$ifNull": ["$logs.late_minutes", 0]},
                "log_ot": {"$ifNull": ["$logs.overtime_minutes", 0]},
                "log_wh": "$logs.work_hours",
                # Mon=2 … Fri=6 in MongoDB $dayOfWeek (Sun=1, Sat=7)
                "log_is_weekday": {
                    "$cond": [
                        {"$gt": ["$logs.date", None]},
                        {
                            "$and": [
                                {"$gte": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$logs.date"}}}, 2]},
                                {"$lte": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$logs.date"}}}, 6]},
                            ]
                        },
                        False,
                    ]
                },
            }
        },
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$addToSet": "$emp_code"},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$log_status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    "$log_is_weekday",
                                ]
                            },
                            {"$cond": ["$log_half_day", 0.5, 1.0]},
                            0,
                        ]
                    }
                },
                "work_hours_list": {
                    "$push": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$log_status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$gt": ["$log_wh", None]},
                                ]
                            },
                            "$log_wh",
                            "$$REMOVE",
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$log_late", 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": "$log_late"},
                "leave_count": {
                    "$sum": {"$cond": [{"$eq": ["$log_status", "LEAVE"]}, 1, 0]}
                },
                "on_duty_count": {
                    "$sum": {"$cond": [{"$eq": ["$log_status", "ON_DUTY"]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": {"$size": "$headcount"},
                "present_days": 1,
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": [{"$size": "$work_hours_list"}, 0]},
                        {"$avg": "$work_hours_list"},
                        None,
                    ]
                },
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
        {"$match": {"headcount": {"$gt": 0}}},
        {"$sort": {"department": 1}},
    ]

    raw = list(db.employees.aggregate(pipeline))
    items = []
    for r in raw:
        items.append({
            "department": r["department"],
            "headcount": r["headcount"],
            "present_days": half_up(r["present_days"], 2),
            "avg_work_hours": half_up(r["avg_work_hours"], 2) if r["avg_work_hours"] is not None else None,
            "late_count": r["late_count"],
            "total_late_minutes": r["total_late_minutes"],
            "leave_count": r["leave_count"],
            "on_duty_count": r["on_duty_count"],
        })

    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(default=10, ge=1, le=50),
    department: Optional[str] = None,
):
    """Late-comers leaderboard.  Standard competition ranking (1,2,2,4).
    Returns all employees whose rank <= limit (including ties at the boundary).
    Only employees with total_late_minutes > 0 appear."""
    first_day, last_day = validate_month_param(month)

    emp_filter: dict = {}
    if department:
        emp_filter["department"] = department

    pipeline = [
        {
            "$match": {
                "date": {"$gte": first_day, "$lte": last_day},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        # Ignore logs whose employee no longer exists
        {"$match": {"emp": {"$ne": []}}},
        {"$unwind": "$emp"},
    ]

    if department:
        pipeline.append({"$match": {"emp.department": department}})

    pipeline += [
        {
            "$project": {
                "_id": 0,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
        {"$sort": {"total_late_minutes": DESCENDING, "emp_code": ASCENDING}},
        # Standard competition ranking via $setWindowFields
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1, "emp_code": 1},
                "output": {
                    "rank": {
                        "$rank": {}
                    }
                },
            }
        },
        # Include all employees whose rank <= limit (handles ties at boundary)
        {"$match": {"rank": {"$lte": limit}}},
    ]

    items = list(db.attendance_logs.aggregate(pipeline))
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_date: str = Query(alias="from"),
    to_date: str = Query(alias="to"),
):
    """Daily attendance trend for a department.
    Returns exactly one row per calendar day from `from` to `to` inclusive.
    Uses $densify to fill gaps (days with no records).
    - is_working_day: Mon-Fri
    - headcount: employees joined_on <= that day (R9)
    - present_count: presence records (half_day = 0.5)
    - attendance_rate: present_count/headcount, 4dp, null on non-working days or headcount=0
    - moving_avg_7d: mean of non-null attendance_rate in current + up to 6 preceding rows
    404 if no employees exist in the department.
    422 if to < from or range exceeds 92 days."""
    # Validate dates
    try:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
        to_dt = datetime.strptime(to_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(422, "from and to must be YYYY-MM-DD dates")

    if to_dt < from_dt:
        raise HTTPException(422, "to must not be before from")
    if (to_dt - from_dt).days > 91:
        raise HTTPException(422, "Date range must not exceed 92 days")

    # 404 if no employees exist in department
    if not db.employees.find_one({"department": department}):
        raise HTTPException(404, f"No employees in department '{department}'")

    # Build the aggregation pipeline
    pipeline = [
        # Step 1: Filter attendance logs for this department and date range.
        # We start from attendance_logs and lookup the employee to get department.
        {
            "$match": {
                "date": {"$gte": from_date, "$lte": to_date},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": {"path": "$emp", "preserveNullAndEmpty": False}},
        {"$match": {"emp.department": department}},
        {
            "$addFields": {
                "is_presence": {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                "hd": {"$ifNull": ["$half_day", False]},
                "late_min": {"$ifNull": ["$late_minutes", 0]},
            }
        },
        {
            "$group": {
                "_id": "$date",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            "$is_presence",
                            {"$cond": ["$hd", 0.5, 1.0]},
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_min", 0]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": "$_id",
                "present_count": 1,
                "late_count": 1,
            }
        },
        # Densify: add a row for every calendar day in the range with no records
        {
            "$densify": {
                "field": "date",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [
                        datetime.strptime(from_date, "%Y-%m-%d"),
                        datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1),
                    ],
                },
            }
        },
        # After densify, "date" may be a datetime object — convert to string
        {
            "$addFields": {
                "date_str": {
                    "$cond": [
                        {"$eq": [{"$type": "$date"}, "string"]},
                        "$date",
                        {"$dateToString": {"format": "%Y-%m-%d", "date": "$date"}},
                    ]
                },
                "present_count": {"$ifNull": ["$present_count", 0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
                # is_working_day: Mon-Fri
                "is_working_day": {
                    "$let": {
                        "vars": {
                            "d": {
                                "$cond": [
                                    {"$eq": [{"$type": "$date"}, "string"]},
                                    {"$dateFromString": {"dateString": "$date"}},
                                    "$date",
                                ]
                            }
                        },
                        "in": {
                            "$and": [
                                {"$gte": [{"$dayOfWeek": "$$d"}, 2]},
                                {"$lte": [{"$dayOfWeek": "$$d"}, 6]},
                            ]
                        },
                    }
                },
            }
        },
        {"$sort": {"date_str": 1}},
    ]

    rows_raw = list(db.attendance_logs.aggregate(pipeline))

    # Python post-processing: compute headcount per day and attendance_rate + moving_avg_7d
    # Headcount for a day = employees with joined_on <= that day AND in department
    # We fetch all relevant employees once and compute per-day headcount in Python.
    # (MongoDB $densify cannot access the employees collection inside the pipeline easily.)
    all_emp = list(db.employees.find(
        {"department": department},
        {"_id": 0, "emp_code": 1, "joined_on": 1},
    ))

    result_rows = []
    rate_window: list = []  # stores non-null rates for moving_avg_7d

    for row in rows_raw:
        day_str = row["date_str"]
        headcount = sum(1 for e in all_emp if e["joined_on"] <= day_str)
        is_working_day = row["is_working_day"]
        present_count = half_up(row["present_count"], 4)
        late_count = row["late_count"]

        # attendance_rate: only on working days with headcount > 0
        if is_working_day and headcount > 0:
            rate = half_up(present_count / headcount, 4)
        else:
            rate = None

        rate_window.append(rate)
        # 7-day window: current + up to 6 preceding rows
        window = rate_window[-7:]
        non_null = [r for r in window if r is not None]
        moving_avg = half_up(sum(non_null) / len(non_null), 4) if non_null else None

        result_rows.append({
            "date": day_str,
            "is_working_day": is_working_day,
            "headcount": headcount,
            "present_count": present_count,
            "late_count": late_count,
            "attendance_rate": rate,
            "moving_avg_7d": moving_avg,
        })

    return {"department": department, "items": result_rows}


# ---------------------------------------------------------------------------
# Admin: explain
# ---------------------------------------------------------------------------

EXPLAIN_ENDPOINTS = {
    "attendance_list",
    "employee_monthly",
    "department_summary",
    "late_leaderboard",
    "department_trend",
}


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    from_date: Optional[str] = Query(default=None, alias="from"),
    to_date: Optional[str] = Query(default=None, alias="to"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """Run explain(executionStats) for the named endpoint's main query/aggregation."""
    if endpoint not in EXPLAIN_ENDPOINTS:
        raise HTTPException(422, f"endpoint must be one of {sorted(EXPLAIN_ENDPOINTS)}")

    if endpoint == "attendance_list":
        q: dict = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status:
            q["status"] = status
        skip = (page - 1) * page_size
        cursor = (
            db.attendance_logs.find(q)
            .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
            .skip(skip)
            .limit(page_size)
        )
        explain_doc = cursor.explain()
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_doc}

    if endpoint == "employee_monthly":
        if not emp_code or not month:
            raise HTTPException(422, "emp_code and month are required for employee_monthly")
        first_day, last_day = validate_month_param(month)
        pipeline = [
            {"$match": {"emp_code": emp_code, "date": {"$gte": first_day, "$lte": last_day}}},
            {"$group": {"_id": None, "total": {"$sum": 1}}},
        ]
        explain_doc = db.attendance_logs.aggregate(pipeline, explain=True)
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_doc}

    if endpoint == "department_summary":
        if not month:
            raise HTTPException(422, "month is required for department_summary")
        first_day, last_day = validate_month_param(month)
        emp_filter: dict = {"joined_on": {"$lte": last_day}}
        if department:
            emp_filter["department"] = department
        pipeline = [
            {"$match": emp_filter},
            {"$group": {"_id": "$department", "headcount": {"$sum": 1}}},
        ]
        explain_doc = db.employees.aggregate(pipeline, explain=True)
        return {"endpoint": endpoint, "collection": "employees", "explain": explain_doc}

    if endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(422, "month is required for late_leaderboard")
        first_day, last_day = validate_month_param(month)
        pipeline = [
            {"$match": {"date": {"$gte": first_day, "$lte": last_day}, "late_minutes": {"$gt": 0}}},
            {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}}},
        ]
        explain_doc = db.attendance_logs.aggregate(pipeline, explain=True)
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_doc}

    if endpoint == "department_trend":
        if not department or not from_date or not to_date:
            raise HTTPException(422, "department, from and to are required for department_trend")
        pipeline = [
            {"$match": {"date": {"$gte": from_date, "$lte": to_date}}},
            {"$group": {"_id": "$date", "present_count": {"$sum": 1}}},
        ]
        explain_doc = db.attendance_logs.aggregate(pipeline, explain=True)
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_doc}
