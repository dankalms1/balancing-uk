from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo

UK = ZoneInfo("Europe/London")

def settlement_periods_in_day(d: date) -> int:
    dt0 = datetime(d.year, d.month, d.day, 0, 0, tzinfo=UK)
    dt1 = dt0 + timedelta(days=1)
    return int((dt1 - dt0).total_seconds() // 1800)

def iso_from_to_for_day(d: date) -> tuple[str,str]:
    iso = d.isoformat()
    return f"{iso}T00:00:00Z", f"{iso}T23:59:59Z"
