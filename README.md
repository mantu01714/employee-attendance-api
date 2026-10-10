# Employee Attendance & Analytics API

A FastAPI + MongoDB REST API for recording employee attendance, computing derived metrics, and producing department-level analytics.

## Prerequisites

- Python 3.11 or 3.12
- MongoDB 6.0 or newer (local, Atlas M0 free tier, or Atlas shared cluster)

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-directory>

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` and set the values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string. For Atlas use `mongodb+srv://...`. |
| `MONGO_DB` | `attendance_db` | Database name. |

**Never commit `.env` or any file containing credentials.**

## MongoDB setup

Indexes are created automatically at application startup (idempotent). No manual setup is needed beyond having a running MongoDB instance.

To load the nine sample documents:

```bash
python sample_seed.py
```

## Running the server

```bash
uvicorn app.main:app --port 8000 --reload
```

The server will be available at `http://localhost:8000`.

## API documentation

Once the server is running:

- Interactive docs (Swagger UI): http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json
- ReDoc: http://localhost:8000/redoc

## Running tests

```bash
pytest tests/ -v
```

Tests require a running MongoDB instance (the same one configured in `.env`). They create and clean up their own data.

## Known limitations

- The `$densify` aggregation stage in `/analytics/departments/{department}/trend` requires MongoDB 5.1 or newer. The assignment specifies MongoDB 6.0, so this is fine.
- The `$setWindowFields` / `$rank` stage for the leaderboard also requires MongoDB 5.0+.
- Headcount computation in the trend endpoint is done in Python (one query per response, not per row) to avoid a per-day `$lookup` inside the densified pipeline; this is efficient for the 92-day maximum range.
- The `moving_avg_7d` window only considers rows within the requested date range (`from` to `to`); rows before `from` are not included, so the first rows use a shorter window as specified.
