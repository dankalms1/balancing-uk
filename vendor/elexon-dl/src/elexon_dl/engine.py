from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from itertools import product
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Mapping, Optional

from dateutil.parser import isoparse

from .http import AsyncHTTP
from .dates import settlement_periods_in_day, iso_from_to_for_day

RowList = List[Mapping[str, Any]]
RowFilter = Callable[[RowList, Dict[str, Any]], RowList]
Enricher  = Callable[[RowList, Dict[str, Any]], RowList]

@dataclass
class TimeStrategy:
    kind: str
    publish_slots: Optional[List[str]] = None
    slot_to_sp: Optional[Dict[str,int]] = None

@dataclass
class EndpointSpec:
    name: str
    method: str = "GET"
    path_template: Optional[str] = None
    query_template: Dict[str, str] = field(default_factory=dict)
    dims: Dict[str, Iterable[Any]] = field(default_factory=dict)
    time: TimeStrategy = field(default_factory=lambda: TimeStrategy(kind="date_only"))
    items_path: Optional[str] = "data"
    table: str = "data"
    primary_keys: Iterable[str] = field(default_factory=tuple)
    row_filter: Optional[RowFilter] = None
    enricher: Optional[Enricher] = None

def _dig(obj, path: Optional[str]):
    if not path:
        return obj
    cur = obj
    for p in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            return None
    return cur

def _items_from_payload(payload: Mapping[str, Any], items_path: Optional[str]) -> RowList:
    if items_path is None:
        items = payload.get("data") or payload.get("items") or payload.get("results")
    else:
        items = _dig(payload, items_path)
    if items is None:
        return []
    if isinstance(items, dict):
        items = list(items.values())
    return list(items)

def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    else: dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

class SpecCrawler:
    def __init__(self, settings, spec: EndpointSpec, *, batch_size: Optional[int] = None):
        self.s = settings
        self.spec = spec
        self.batch_size = batch_size or (self.s.max_concurrency * 8)

    def _contexts_for_day(self, d: date) -> List[Dict[str, Any]]:
        t = self.spec.time
        ctxs: List[Dict[str, Any]] = []
        dims_items = list(self.spec.dims.items())
        dim_keys = [k for k,_ in dims_items]
        dim_values_product = list(product(*[list(vs) for _,vs in dims_items])) or [()]

        if t.kind == "date_sp":
            nsp = settlement_periods_in_day(d)
            for sp in range(1, nsp+1):
                base = {"date": d.isoformat(), "sp": sp}
                for values in dim_values_product:
                    ctx = dict(base); ctx.update(dict(zip(dim_keys, values)))
                    ctxs.append(ctx)
        elif t.kind == "date_only":
            base = {"date": d.isoformat()}
            for values in dim_values_product:
                ctx = dict(base); ctx.update(dict(zip(dim_keys, values)))
                ctxs.append(ctx)
        elif t.kind == "from_to":
            from_ts, to_ts = iso_from_to_for_day(d)
            base = {"date": d.isoformat(), "from_ts": from_ts, "to_ts": to_ts}
            for values in dim_values_product:
                ctx = dict(base); ctx.update(dict(zip(dim_keys, values)))
                ctxs.append(ctx)
        elif t.kind == "publish_slots":
            for slot in (t.publish_slots or []):
                pub = datetime.fromisoformat(f"{d.isoformat()}T{slot}:00+00:00").astimezone(timezone.utc)
                base = {"date": d.isoformat(), "publishTime": _iso_z(pub)}
                if t.slot_to_sp and slot in t.slot_to_sp:
                    base["slot_sp"] = t.slot_to_sp[slot]
                for values in dim_values_product:
                    ctx = dict(base); ctx.update(dict(zip(dim_keys, values)))
                    ctxs.append(ctx)
        elif t.kind == "halfhour_slots":
            cur = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            end = cur + timedelta(days=1)
            while cur < end:
                base = {"date": d.isoformat(), "publishTime": _iso_z(cur)}
                for values in dim_values_product:
                    ctx = dict(base); ctx.update(dict(zip(dim_keys, values)))
                    ctxs.append(ctx)
                cur += timedelta(minutes=30)
        else:
            raise ValueError(f"Unknown time strategy: {t.kind}")
        return ctxs

    def _build_request(self, ctx: Dict[str, Any], extra_params: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        base = self.s.base_url.rstrip("/")
        if self.spec.path_template:
            path = self.spec.path_template.format_map(ctx)
            url = f"{base}{path}"
        else:
            url = base
        params = {k: (v.format_map(ctx) if isinstance(v, str) else v) for k,v in self.spec.query_template.items()}
        params.update(extra_params or {})
        return url, params

    def _enrich(self, rows: RowList, ctx: Dict[str, Any]) -> RowList:
        out: RowList = []
        for r in rows:
            rec = dict(r)
            for k in ("date","sp"):
                if k in ctx:
                    name = "settlementPeriod" if k=="sp" else "date"
                    rec.setdefault(name, ctx[k])
            for k,v in ctx.items():
                if k not in ("date","sp","publishTime","from_ts","to_ts","slot_sp"):
                    rec.setdefault(k, v)
            if "slot_sp" in ctx:
                rec.setdefault("settlementPeriod", ctx["slot_sp"])
            out.append(rec)
        if self.spec.enricher:
            out = self.spec.enricher(out, ctx)
        return out

    async def _fetch_ctx(self, http: AsyncHTTP, ctx: Dict[str, Any], extra_params: Dict[str, Any]) -> RowList:
        url, params = self._build_request(ctx, extra_params)
        r = await http.get(url, params=params)
        if r.status_code in (204,404):
            return []
        r.raise_for_status()
        payload = r.json()
        rows = _items_from_payload(payload, self.spec.items_path)
        if not rows:
            return []
        rows = self._enrich(rows, ctx)
        if self.spec.row_filter:
            rows = self.spec.row_filter(rows, ctx)
        return rows

    async def pages(self, http: AsyncHTTP, *, start_date: date, end_date: date, **extra_params) -> AsyncIterator[RowList]:
        sem = asyncio.Semaphore(self.s.max_concurrency)
        async def guarded(coro):
            async with sem:
                return await coro

        batch: List[asyncio.Task] = []
        d = start_date
        while d <= end_date:
            for ctx in self._contexts_for_day(d):
                batch.append(asyncio.create_task(guarded(self._fetch_ctx(http, ctx, extra_params))))
                if len(batch) >= self.batch_size:
                    results = await asyncio.gather(*batch)
                    batch.clear()
                    chunk = [row for rows in results for row in rows]
                    if chunk:
                        yield chunk
            d = d + timedelta(days=1)
        if batch:
            results = await asyncio.gather(*batch)
            chunk = [row for rows in results for row in rows]
            if chunk:
                yield chunk
