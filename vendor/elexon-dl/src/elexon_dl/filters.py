
from __future__ import annotations
from typing import Any, Dict, List, Mapping
from datetime import datetime, timedelta, timezone
from dateutil.parser import isoparse

RowList = List[Mapping[str, Any]]

def exact_sp_filter(rows: RowList, ctx: Dict[str, Any]) -> RowList:
    """Keep rows matching the exact settlement period (from "settlementPeriod" or "settlementPeriodFrom")."""
    sp = int(ctx.get("sp", ctx.get("settlementPeriod", -1)))
    out = []
    for it in rows:
        if "settlementPeriod" in it and int(it["settlementPeriod"]) != sp:
            continue
        sp_from = it.get("settlementPeriodFrom")
        if sp_from is not None and int(sp_from) != sp:
            continue
        out.append(it)
    return out

def within_dayahead_window(rows: RowList, ctx: Dict[str, Any]) -> RowList:
    """Keep rows with startTime in [publishTimeEffective+30m, publishTimeEffective+24h]."""
    pub_eff_raw = ctx.get("publishTimeEffective") or ctx.get("publishTime")
    if pub_eff_raw is None:
        return []
    if isinstance(pub_eff_raw, str):
        pub = isoparse(pub_eff_raw).astimezone(timezone.utc)
    else:
        pub = pub_eff_raw.astimezone(timezone.utc)
    window_start = pub + timedelta(minutes=30)
    window_end = pub + timedelta(hours=24)
    out = []
    for it in rows:
        st_raw = it.get("startTime")
        try:
            st = isoparse(st_raw).astimezone(timezone.utc) if st_raw else None
        except Exception:
            st = None
        if st is None: 
            continue
        if window_start <= st <= window_end:
            out.append(it)
    return out

def enrich_publish_effective(rows: RowList, ctx: Dict[str, Any]) -> RowList:
    """Add publishTimeEffective (do not overwrite publishTime) based on payload publishTime or query param."""
    pub_ctx = ctx.get("publishTime")
    if pub_ctx is None:
        return rows
    try:
        pub_eff = isoparse(rows[0].get("publishTime","")).astimezone(timezone.utc)
    except Exception:
        pub_eff = isoparse(pub_ctx).astimezone(timezone.utc)
    iso = pub_eff.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for it in rows:
        rec = dict(it)
        rec.setdefault("publishTimeEffective", iso)
        out.append(rec)
    ctx["publishTimeEffective"] = iso
    return out

def wind_evolution_top8(rows: RowList, ctx: Dict[str, Any]) -> RowList:
    """Keep publishes <= startTime-1h, take latest 8 by publishTime."""
    start_raw = ctx.get("publishTime")  # here used as startTime
    if start_raw is None:
        return []
    start_dt = isoparse(start_raw).astimezone(timezone.utc)
    cutoff = start_dt - timedelta(hours=1)
    def parse_pub(r):
        try:
            return isoparse(r.get("publishTime","")).astimezone(timezone.utc)
        except Exception:
            return None
    filtered = [r for r in rows if ((parse_pub(r) or start_dt) <= cutoff)]
    filtered.sort(key=lambda r: parse_pub(r) or datetime(1970,1,1, tzinfo=timezone.utc), reverse=True)
    return filtered[:8]
