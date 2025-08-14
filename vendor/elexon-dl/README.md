# elexon-dl

Spec-driven async Elexon/BMRS crawler. One engine + tiny endpoint specs. Handles:
- date×SP
- date-only
- from/to windows
- publish-time slots
- half-hour slots

Default output is JSONL. CSV/Parquet optional.

## Quick start

```bash
pip install -U pip
pip install -e .[dev,parquet]  # dev extras give you ruff/mypy/pytest
elexon-dl --help
```

## Examples

```bash
# Health check (non-zero exit code on unhealthy)
elexon-dl health

# Crawl wind history publishes for a date range to JSONL
elexon-dl crawl --spec wind_history --start-date 2024-12-01 --end-date 2024-12-02 --output-dir data

# Same to CSV
elexon-dl crawl --spec wind_history --start-date 2024-12-01 --end-date 2024-12-02 --output-dir data --format csv
```

## Development

```bash
make venv
make lint
make typecheck
make test
make build
```

## Docker

```bash
docker build -t elexon-dl:latest .
docker run --rm -v $PWD/data:/app/data elexon-dl:latest   elexon-dl crawl --spec system_prices --start-date 2024-12-01 --end-date 2024-12-03
```
