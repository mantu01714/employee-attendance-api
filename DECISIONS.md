# DECISIONS.md

1. **Indexes.** I created three indexes at startup.

   - `employees.emp_code` (unique): enforces the business rule that `emp_code` is unique and never changes. This lets the unique-index + `DuplicateKeyError` pattern handle concurrent duplicate creates safely without a check-then-insert race. It also makes every employee look-up O(1) by `emp_code`.
   - `attendance_logs(emp_code ASC, date ASC)` (unique): the natural key of an attendance record. A record is addressed by `(emp_code, date)` everywhere in the API; the unique constraint prevents duplicate punch-ins even under concurrency. I considered a separate index on just `date` but the compound one covers both the look-up pattern and the uniqueness requirement.
   - `attendance_logs(date DESC, emp_code ASC)`: serves the `/attendance` list sort order exactly, so MongoDB can satisfy the sort with an IXSCAN rather than an in-memory sort, which is required for the 100k-record dataset.

   Index I considered and rejected: a separate index on `attendance_logs.emp_code` alone. The compound `(emp_code, date)` index already covers prefix queries on `emp_code`, so a standalone `emp_code` index would be redundant.

2. **Punch-in race.** Two identical punch-in requests arrive at the same instant. Both requests read the employee record (no race there). Both attempt `insert_one` with the same `(emp_code, date)`. MongoDB's unique compound index allows only one insert to succeed; the other receives a `DuplicateKeyError`. The winning request returns 201; the losing request's `except DuplicateKeyError` block raises `HTTPException(409)`. No application-level lock is needed.

3. **Ties.** Two employees with identical `total_late_minutes` at the cut-off rank are both included in the response. The `$setWindowFields` + `$rank` stage assigns the same rank to tied employees (standard competition ranking: 1, 2, 2, 4). The final `$match` keeps every row whose rank ≤ `limit`. So if rank 10 is a four-way tie, all four appear even though the response has more than `limit` rows. This matches the spec: "return every employee whose rank ≤ limit."

4. **Headcount.** The department summary pipeline starts from the `employees` collection (not `attendance_logs`) and uses `$lookup` + `preserveNullAndEmpty: True` to keep employees with zero logs. Employees are counted using `$addToSet` on `emp_code` (deduplicated) within the `$group` stage, then `$size` on that set. This ensures every employee whose `joined_on ≤ last day of the month` is counted even if they have no attendance records in that month.

5. **One thing I would change at 100× the data.** Move the working-days calculation and the per-day headcount in the trend endpoint fully into MongoDB aggregation instead of computing them in Python. At 100× scale the 92-day loop and the employee fetch-and-iterate would become noticeable. I would pre-aggregate employees per department per day using a `$densify` + `$lookup` pipeline in MongoDB, eliminating the Python post-processing loop and reducing round-trips to one.
