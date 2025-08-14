import asyncio
from pathlib import Path
from datetime import date
from typing import Optional, List

import typer

from .config import Settings
from .http import AsyncHTTP
from .storage import JSONStore, CSVStore, ParquetStore
from .engine import SpecCrawler
from .specs import SPEC_REGISTRY
from .progress import ProgressReporter
from .health import api_health

app = typer.Typer(add_completion=False, no_args_is_help=True)

def _make_store(output_dir: Path, fmt: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        return CSVStore(output_dir)
    if fmt == "parquet":
        return ParquetStore(output_dir)
    return JSONStore(output_dir)  # default

@app.command()
def health():
    s = Settings()
    async def _run():
        async with AsyncHTTP(s) as http:
            res = await api_health(http, s)
            import json
            print(json.dumps(res, indent=2))
            if not res.get("_ok"):
                raise typer.Exit(code=1)
    asyncio.run(_run())

@app.command()
def crawl(
    spec: str = typer.Option(..., help="Spec name (see specs.py)"),
    start_date: str = typer.Option(..., help="YYYY-MM-DD"),
    end_date: str = typer.Option(..., help="YYYY-MM-DD"),
    output_dir: Path = typer.Option(Path("data"), help="Output directory"),
    format: str = typer.Option("json", help="json|csv|parquet"),
    progress: bool = typer.Option(False, help="Show live progress table"),
    params: List[str] = typer.Argument(None, help="Extra query params as key=value (overrides spec defaults)"),
):
    if spec not in SPEC_REGISTRY:
        raise typer.BadParameter(f"Unknown spec '{spec}'. Available: {', '.join(sorted(SPEC_REGISTRY))}")

    s = Settings()
    sd = date.fromisoformat(start_date)
    ed = date.fromisoformat(end_date)
    store = _make_store(output_dir, format)
    extra = {}
    for kv in params or []:
        if "=" not in kv:
            raise typer.BadParameter(f"Bad param format: {kv}, expected key=value")
        k, v = kv.split("=", 1)
        extra[k] = v

    async def _run():
        async with AsyncHTTP(s) as http:
            pr = ProgressReporter(http) if progress else None
            if pr: pr.start()
            crawler = SpecCrawler(s, SPEC_REGISTRY[spec])
            total = 0
            async for chunk in crawler.pages(http, start_date=sd, end_date=ed, **extra):
                store.upsert(SPEC_REGISTRY[spec].table, chunk, keys=list(SPEC_REGISTRY[spec].primary_keys))
                total += len(chunk)
            if pr: pr.stop()
            typer.echo(f"Wrote {total} rows to {output_dir} ({SPEC_REGISTRY[spec].table}.{format})")

    asyncio.run(_run())
