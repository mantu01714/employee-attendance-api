"""
tests/test_api.py — Integration tests for the Employee Attendance & Analytics API.

These tests require a running MongoDB instance (configured via MONGO_URI / MONGO_DB
in .env or environment variables).  Each test cleans up its own data.

Run:  pytest tests/ -v
"""
import math
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

# Ensure we load environment variables before importing app.main
from dotenv import load_dotenv
load_dotenv()

from app.main import app, db, IST, half_up, ms_to_dt, dt_to_ms, attendance_date, compute_late_minutes

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
IST_TZ = timezone(timedelta(hours=5, minutes=30))

def ist_ms(year, month, day, hour, minute, second=0):
    """Return epoch milliseconds for a given IST datetime."""
    dt = datetime(year, month, day, hour, minute, second, tzinfo=IST_TZ)
    return int(dt.timestamp() * 1000)


def now_ms():
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def cleanup_emp(emp_code: str):
    db.employees.delete_many({"emp_code": emp_code})
    db.attendance_logs.delete_many({"emp_code": emp_code})


EMP_PAYLOAD = {
    "emp_code": "EMPTEST1",
    "name": "Test User",
    "email": "test@example.com",
    "department": "Testing",
    "shift_start": "09:30",
    "shift_end": "18:30",
    "joined_on": "2026-01-01",
}

OVERNIGHT_EMP_PAYLOAD = {
    "emp_code": "EMPTEST2",
    "name": "Night Worker",
    "email": "night@example.com",
    "department": "Testing",
    "shift_start": "22:00",
    "shift_end": "06:00",
    "joined_on": "2026-01-01",
}


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test employees before and after each test."""
    cleanup_emp("EMPTEST1")
    cleanup_emp("EMPTEST2")
    yield
    cleanup_emp("EMPTEST1")
    cleanup_emp("EMPTEST2")


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_half_up_rounding(self):
        """Python round() uses banker's rounding; half_up must always round 0.5 up."""
        assert half_up(2.125, 2) == 2.13   # banker's would give 2.12
        assert half_up(2.135, 2) == 2.14
        assert half_up(9.085, 2) == 9.09
        assert half_up(0.5, 0) == 1

    def test_ms_to_dt_truncates_microseconds(self):
        """R1: instants are truncated to whole seconds."""
        ms = 1783312500999  # has sub-second component
        dt = ms_to_dt(ms)
        assert dt.microsecond == 0
        assert dt.tzinfo is not None  # must be aware

    def test_attendance_date_normal(self):
        """Normal shift: IST calendar date of punch_in."""
        # 09:28 IST on 2026-07-06 = 03:58 UTC
        punch = datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc)
        date = attendance_date(punch, "09:30", "18:30")
        assert date == "2026-07-06"

    def test_attendance_date_after_midnight_ist(self):
        """00:30 IST = 19:00 UTC previous day; attendance date should be IST date."""
        punch = datetime(2026, 7, 5, 19, 0, 0, tzinfo=timezone.utc)  # 00:30 IST on July 6
        date = attendance_date(punch, "09:30", "18:30")
        assert date == "2026-07-06"

    def test_attendance_date_overnight_late_night(self):
        """Overnight shift 22:00-06:00: punch at 23:00 IST belongs to that day."""
        punch = datetime(2026, 7, 6, 17, 30, 0, tzinfo=timezone.utc)  # 23:00 IST on July 6
        date = attendance_date(punch, "22:00", "06:00")
        assert date == "2026-07-06"

    def test_attendance_date_overnight_early_morning(self):
        """Overnight shift 22:00-06:00: punch at 03:00 IST on July 7 belongs to July 6."""
        punch = datetime(2026, 7, 6, 21, 30, 0, tzinfo=timezone.utc)  # 03:00 IST on July 7
        date = attendance_date(punch, "22:00", "06:00")
        assert date == "2026-07-06"

    def test_late_minutes_grace_period(self):
        """09:40:00 IST = exactly 10 min late → NOT late (must be strictly > 10 min)."""
        punch = datetime(2026, 7, 6, 4, 10, 0, tzinfo=timezone.utc)  # 09:40 IST
        late = compute_late_minutes(punch, "09:30", "2026-07-06")
        assert late == 0

    def test_late_minutes_one_second_over_grace(self):
        """09:40:01 IST = 10 min 1 sec late → floor = 10 minutes."""
        punch = datetime(2026, 7, 6, 4, 10, 1, tzinfo=timezone.utc)  # 09:40:01 IST
        late = compute_late_minutes(punch, "09:30", "2026-07-06")
        assert late == 10

    def test_late_minutes_45(self):
        """10:15:59 IST → 45 minutes late (spec example)."""
        punch = datetime(2026, 7, 6, 4, 45, 59, tzinfo=timezone.utc)  # 10:15:59 IST
        late = compute_late_minutes(punch, "09:30", "2026-07-06")
        assert late == 45


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /employees — POST (create)
# ---------------------------------------------------------------------------

class TestCreateEmployee:
    def test_create_ok(self):
        r = client.post("/employees", json=EMP_PAYLOAD)
        assert r.status_code == 201
        data = r.json()
        assert data["emp_code"] == "EMPTEST1"
        assert isinstance(data["created_at"], int)
        assert data["created_at"] > 100_000_000_000  # epoch ms

    def test_duplicate_emp_code_409(self):
        client.post("/employees", json=EMP_PAYLOAD)
        r = client.post("/employees", json=EMP_PAYLOAD)
        assert r.status_code == 409

    def test_invalid_emp_code_format_422(self):
        bad = {**EMP_PAYLOAD, "emp_code": "E001"}
        r = client.post("/employees", json=bad)
        assert r.status_code == 422

    def test_shift_start_equals_shift_end_422(self):
        bad = {**EMP_PAYLOAD, "shift_start": "09:30", "shift_end": "09:30"}
        r = client.post("/employees", json=bad)
        assert r.status_code == 422

    def test_invalid_email_422(self):
        bad = {**EMP_PAYLOAD, "email": "notanemail"}
        r = client.post("/employees", json=bad)
        assert r.status_code == 422

    def test_overnight_shift_accepted(self):
        r = client.post("/employees", json=OVERNIGHT_EMP_PAYLOAD)
        assert r.status_code == 201

    def test_no_id_field_in_response(self):
        r = client.post("/employees", json=EMP_PAYLOAD)
        assert "_id" not in r.json()
        assert "id" not in r.json()


# ---------------------------------------------------------------------------
# /employees — GET (list)
# ---------------------------------------------------------------------------

class TestListEmployees:
    def setup_method(self):
        client.post("/employees", json=EMP_PAYLOAD)

    def test_list_returns_created_employee(self):
        r = client.get("/employees")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert "total" in data
        codes = [e["emp_code"] for e in data["items"]]
        assert "EMPTEST1" in codes

    def test_department_filter_total_is_filtered(self):
        r = client.get("/employees", params={"department": "Testing"})
        data = r.json()
        assert data["total"] == len(data["items"])
        for item in data["items"]:
            assert item["department"] == "Testing"

    def test_page_starts_at_1(self):
        r1 = client.get("/employees", params={"page": 1, "page_size": 1})
        r2 = client.get("/employees", params={"page": 2, "page_size": 1})
        d1 = r1.json()["items"]
        d2 = r2.json()["items"]
        if d1 and d2:
            assert d1[0]["emp_code"] != d2[0]["emp_code"]

    def test_page_size_over_100_422(self):
        r = client.get("/employees", params={"page_size": 101})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /attendance/punch-in
# ---------------------------------------------------------------------------

class TestPunchIn:
    def setup_method(self):
        client.post("/employees", json=EMP_PAYLOAD)

    def test_punch_in_ok(self):
        punch_ms = ist_ms(2026, 7, 6, 9, 28, 0)
        r = client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": punch_ms})
        assert r.status_code == 201
        data = r.json()
        assert data["emp_code"] == "EMPTEST1"
        assert data["date"] == "2026-07-06"
        assert data["status"] == "PRESENT"
        assert data["punch_in"] == punch_ms
        assert data["punch_out"] is None
        assert data["work_hours"] is None
        assert data["late_minutes"] == 0  # 09:28, within grace
        assert data["overtime_minutes"] == 0
        assert data["half_day"] is False
        assert data["history"] == []

    def test_punch_in_late(self):
        punch_ms = ist_ms(2026, 7, 6, 10, 5, 0)  # 35 min late
        r = client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": punch_ms})
        assert r.status_code == 201
        assert r.json()["late_minutes"] == 35

    def test_punch_in_grace_boundary(self):
        """09:40:00 → NOT late (spec example)."""
        punch_ms = ist_ms(2026, 7, 6, 9, 40, 0)
        r = client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": punch_ms})
        assert r.status_code == 201
        assert r.json()["late_minutes"] == 0

    def test_duplicate_punch_in_409(self):
        punch_ms = ist_ms(2026, 7, 6, 9, 28, 0)
        client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": punch_ms})
        r = client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": punch_ms})
        assert r.status_code == 409

    def test_unknown_employee_404(self):
        r = client.post("/attendance/punch-in", json={"emp_code": "EMP9999", "punched_at": ist_ms(2026, 7, 6, 9, 0)})
        assert r.status_code == 404

    def test_absent_status_rejected_422(self):
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMPTEST1",
            "punched_at": ist_ms(2026, 7, 6, 9, 0),
            "status": "ABSENT",
        })
        assert r.status_code == 422

    def test_leave_status_rejected_422(self):
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMPTEST1",
            "punched_at": ist_ms(2026, 7, 6, 9, 0),
            "status": "LEAVE",
        })
        assert r.status_code == 422

    def test_wfh_status_accepted(self):
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMPTEST1",
            "punched_at": ist_ms(2026, 7, 6, 9, 0),
            "status": "WFH",
        })
        assert r.status_code == 201
        assert r.json()["status"] == "WFH"

    def test_seconds_looking_value_422(self):
        """Values below 100_000_000_000 look like seconds and must be rejected."""
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMPTEST1",
            "punched_at": 1783312,  # obviously seconds
        })
        assert r.status_code == 422

    def test_no_id_field_in_response(self):
        punch_ms = ist_ms(2026, 7, 6, 9, 0)
        r = client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": punch_ms})
        assert "_id" not in r.json()
        assert "id" not in r.json()

    def test_attendance_date_is_ist_not_utc(self):
        """Punch at 00:30 IST (19:00 UTC previous day) → IST date, not UTC date."""
        punch_ms = ist_ms(2026, 7, 6, 0, 30, 0)  # 00:30 IST Jul 6 = 19:00 UTC Jul 5
        r = client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": punch_ms})
        assert r.status_code == 201
        assert r.json()["date"] == "2026-07-06"  # must be IST date


# ---------------------------------------------------------------------------
# /attendance/punch-out
# ---------------------------------------------------------------------------

class TestPunchOut:
    def setup_method(self):
        client.post("/employees", json=EMP_PAYLOAD)
        punch_ms = ist_ms(2026, 7, 6, 9, 28, 0)
        client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": punch_ms})
        self.punch_in_ms = punch_ms

    def test_punch_out_ok(self):
        punch_out_ms = ist_ms(2026, 7, 6, 18, 35, 0)  # 9h 7min work, 5 min overtime (<30 so 0)
        r = client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": punch_out_ms})
        assert r.status_code == 200
        data = r.json()
        assert data["punch_out"] == punch_out_ms
        assert data["work_hours"] is not None
        assert data["work_hours"] == half_up(9 * 3600 / 3600 + 7 * 60 / 3600, 2)
        assert data["overtime_minutes"] == 0  # 5 min < 30 threshold
        assert data["half_day"] is False

    def test_punch_out_with_overtime(self):
        punch_out_ms = ist_ms(2026, 7, 6, 19, 10, 0)  # 40 min overtime
        r = client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": punch_out_ms})
        assert r.status_code == 200
        assert r.json()["overtime_minutes"] == 40

    def test_punch_out_half_day(self):
        punch_out_ms = ist_ms(2026, 7, 6, 13, 0, 0)  # 3h 32min work = 3.53h < 4.50 → half day
        r = client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": punch_out_ms})
        assert r.status_code == 200
        data = r.json()
        assert data["half_day"] is True
        assert data["work_hours"] < 4.50

    def test_double_punch_out_409(self):
        punch_out_ms = ist_ms(2026, 7, 6, 18, 30, 0)
        client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": punch_out_ms})
        r = client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": punch_out_ms})
        assert r.status_code == 409

    def test_punch_out_before_punch_in_422(self):
        punch_out_ms = ist_ms(2026, 7, 6, 8, 0, 0)  # before 09:28
        r = client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": punch_out_ms})
        assert r.status_code == 422

    def test_punch_out_more_than_24h_422(self):
        punch_out_ms = ist_ms(2026, 7, 7, 9, 29, 0)  # 24h 1min after punch_in
        r = client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": punch_out_ms})
        assert r.status_code == 422

    def test_unknown_employee_404(self):
        r = client.post("/attendance/punch-out", json={"emp_code": "EMP9999", "punched_at": ist_ms(2026, 7, 6, 18, 30)})
        assert r.status_code == 404

    def test_no_open_punch_in_404(self):
        # Employee has no open record (they haven't punched in)
        r = client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": ist_ms(2026, 7, 7, 18, 30)})
        # There's no record with punch_in <= that time that's open on July 7 (record is on July 6)
        # This is actually valid because it finds the July 6 record...
        # Let's test with a genuinely unknown employee instead
        r2 = client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": ist_ms(2026, 7, 5, 18, 30)})
        assert r2.status_code == 404


# ---------------------------------------------------------------------------
# /attendance (list)
# ---------------------------------------------------------------------------

class TestListAttendance:
    def setup_method(self):
        client.post("/employees", json=EMP_PAYLOAD)
        client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": ist_ms(2026, 7, 6, 9, 28)})
        client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": ist_ms(2026, 7, 7, 9, 28)})

    def test_list_ok(self):
        r = client.get("/attendance")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert "total" in data

    def test_filter_by_emp_code(self):
        r = client.get("/attendance", params={"emp_code": "EMPTEST1"})
        data = r.json()
        for item in data["items"]:
            assert item["emp_code"] == "EMPTEST1"

    def test_sorted_date_desc(self):
        r = client.get("/attendance", params={"emp_code": "EMPTEST1"})
        dates = [item["date"] for item in r.json()["items"]]
        assert dates == sorted(dates, reverse=True)

    def test_date_range_filter(self):
        r = client.get("/attendance", params={"date_from": "2026-07-07", "date_to": "2026-07-07", "emp_code": "EMPTEST1"})
        data = r.json()
        assert data["total"] == 1
        assert data["items"][0]["date"] == "2026-07-07"

    def test_date_from_after_date_to_422(self):
        r = client.get("/attendance", params={"date_from": "2026-07-10", "date_to": "2026-07-01"})
        assert r.status_code == 422

    def test_punch_in_is_epoch_ms(self):
        r = client.get("/attendance", params={"emp_code": "EMPTEST1"})
        item = r.json()["items"][0]
        assert isinstance(item["punch_in"], int)
        assert item["punch_in"] > 100_000_000_000


# ---------------------------------------------------------------------------
# /attendance/{emp_code}/{date} — PATCH (regularize)
# ---------------------------------------------------------------------------

class TestRegularize:
    def setup_method(self):
        client.post("/employees", json=EMP_PAYLOAD)
        # Create a record with late punch-in
        client.post("/attendance/punch-in", json={
            "emp_code": "EMPTEST1",
            "punched_at": ist_ms(2026, 7, 6, 10, 5, 0),  # 35 min late
        })
        client.post("/attendance/punch-out", json={
            "emp_code": "EMPTEST1",
            "punched_at": ist_ms(2026, 7, 6, 18, 35, 0),
        })

    def test_regularize_punch_in_correction(self):
        new_punch_ms = ist_ms(2026, 7, 6, 9, 28, 0)  # no longer late
        r = client.patch("/attendance/EMPTEST1/2026-07-06", json={
            "punch_in": new_punch_ms,
            "reason": "Biometric glitch corrected",
            "regularized_by": "hr.admin",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["punch_in"] == new_punch_ms
        assert data["late_minutes"] == 0  # recomputed
        assert len(data["history"]) == 1
        entry = data["history"][0]
        assert entry["by"] == "hr.admin"
        assert entry["reason"] == "Biometric glitch corrected"
        assert "punch_in" in entry["changes"]
        assert "late_minutes" in entry["changes"]

    def test_history_appended_not_overwritten(self):
        new_punch_ms = ist_ms(2026, 7, 6, 9, 28, 0)
        client.patch("/attendance/EMPTEST1/2026-07-06", json={
            "punch_in": new_punch_ms,
            "reason": "First correction here",
            "regularized_by": "hr.admin",
        })
        newer_punch_ms = ist_ms(2026, 7, 6, 9, 30, 0)
        r = client.patch("/attendance/EMPTEST1/2026-07-06", json={
            "punch_in": newer_punch_ms,
            "reason": "Second correction needed",
            "regularized_by": "hr.manager",
        })
        assert r.status_code == 200
        assert len(r.json()["history"]) == 2  # both entries preserved

    def test_no_change_422(self):
        # Trying to regularize without actually changing anything
        current = client.get("/attendance", params={"emp_code": "EMPTEST1", "date_from": "2026-07-06", "date_to": "2026-07-06"})
        current_punch_in = current.json()["items"][0]["punch_in"]
        r = client.patch("/attendance/EMPTEST1/2026-07-06", json={
            "punch_in": current_punch_in,  # same value → no change
            "reason": "Nothing changed here",
            "regularized_by": "hr.admin",
        })
        assert r.status_code == 422

    def test_absent_with_punch_in_422(self):
        r = client.patch("/attendance/EMPTEST1/2026-07-06", json={
            "status": "ABSENT",
            "punch_in": ist_ms(2026, 7, 6, 9, 0),
            "reason": "Marking absent with punch",
            "regularized_by": "hr.admin",
        })
        assert r.status_code == 422

    def test_unknown_employee_404(self):
        r = client.patch("/attendance/EMP9999/2026-07-06", json={
            "reason": "Correction for ghost",
            "regularized_by": "hr.admin",
        })
        assert r.status_code == 404

    def test_unknown_date_404(self):
        r = client.patch("/attendance/EMPTEST1/2026-01-01", json={
            "reason": "Record does not exist",
            "regularized_by": "hr.admin",
        })
        assert r.status_code == 404

    def test_reason_too_short_422(self):
        r = client.patch("/attendance/EMPTEST1/2026-07-06", json={
            "reason": "No",  # less than 5 chars
            "regularized_by": "hr.admin",
        })
        assert r.status_code == 422

    def test_absent_clears_punch_times(self):
        r = client.patch("/attendance/EMPTEST1/2026-07-06", json={
            "status": "ABSENT",
            "reason": "Employee was absent actually",
            "regularized_by": "hr.admin",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ABSENT"
        assert data["punch_in"] is None
        assert data["punch_out"] is None

    def test_history_changes_punch_times_are_epoch_ms(self):
        new_punch_ms = ist_ms(2026, 7, 6, 9, 28, 0)
        r = client.patch("/attendance/EMPTEST1/2026-07-06", json={
            "punch_in": new_punch_ms,
            "reason": "Biometric glitch fixed here",
            "regularized_by": "hr.admin",
        })
        entry = r.json()["history"][0]
        # from and to for punch_in must be epoch ms integers (not datetimes)
        assert isinstance(entry["changes"]["punch_in"]["from"], int)
        assert isinstance(entry["changes"]["punch_in"]["to"], int)


# ---------------------------------------------------------------------------
# Analytics: /analytics/employees/{emp_code}/monthly
# ---------------------------------------------------------------------------

class TestEmployeeMonthly:
    def setup_method(self):
        client.post("/employees", json=EMP_PAYLOAD)
        # Monday 2026-07-06: present, on time, 9.12h
        client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": ist_ms(2026, 7, 6, 9, 28)})
        client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": ist_ms(2026, 7, 6, 18, 35)})
        # Tuesday 2026-07-07: present, late, short day (half day)
        client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": ist_ms(2026, 7, 7, 10, 5)})
        client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": ist_ms(2026, 7, 7, 13, 0)})

    def test_employee_monthly_ok(self):
        r = client.get("/analytics/employees/EMPTEST1/monthly", params={"month": "2026-07"})
        assert r.status_code == 200
        data = r.json()
        assert data["emp_code"] == "EMPTEST1"
        assert data["month"] == "2026-07"
        assert isinstance(data["working_days"], int)
        assert data["working_days"] > 0
        # Jul 6 = 1.0 present day, Jul 7 half_day so 0.5 → total 1.5
        assert data["present_days"] == 1.5
        assert data["late_count"] == 1  # only Jul 7 is late
        assert data["leave_days"] == 0

    def test_unknown_employee_404(self):
        r = client.get("/analytics/employees/EMP9999/monthly", params={"month": "2026-07"})
        assert r.status_code == 404

    def test_no_logs_returns_zeros(self):
        # New employee with no logs
        client.post("/employees", json={**EMP_PAYLOAD, "emp_code": "EMPTESTZ", "email": "z@example.com"})
        r = client.get("/analytics/employees/EMPTESTZ/monthly", params={"month": "2026-07"})
        db.employees.delete_many({"emp_code": "EMPTESTZ"})
        assert r.status_code == 200
        data = r.json()
        assert data["present_days"] == 0
        assert data["leave_days"] == 0
        assert data["late_count"] == 0
        assert data["attendance_pct"] == 0.0 or data["attendance_pct"] is None


# ---------------------------------------------------------------------------
# Work hours R4: half-up rounding correctness
# ---------------------------------------------------------------------------

class TestWorkHoursRounding:
    def setup_method(self):
        client.post("/employees", json=EMP_PAYLOAD)

    def test_work_hours_half_up(self):
        """(punch_out - punch_in) = 9h 7min = 9.1166... → rounds half-up to 9.12 (spec example ~9.12)."""
        punch_in_ms = ist_ms(2026, 7, 6, 9, 28, 0)
        punch_out_ms = ist_ms(2026, 7, 6, 18, 35, 0)  # 9h 7min
        client.post("/attendance/punch-in", json={"emp_code": "EMPTEST1", "punched_at": punch_in_ms})
        r = client.post("/attendance/punch-out", json={"emp_code": "EMPTEST1", "punched_at": punch_out_ms})
        wh = r.json()["work_hours"]
        # 9*3600 + 7*60 = 32820 seconds / 3600 = 9.11666... → half_up = 9.12
        expected = half_up((punch_out_ms - punch_in_ms) / 1000 / 3600, 2)
        assert wh == expected
