from pathlib import Path
from typing import Iterable, Mapping, Any, Optional, List
import json
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _HAS_PARQUET = True
except Exception:
    _HAS_PARQUET = False

class JSONStore:
    def __init__(self, base: Path):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
    def _path(self, table: str) -> Path:
        return self.base / f"{table}.jsonl"
    def upsert(self, table: str, records: List[Mapping[str, Any]], keys: Optional[List[str]] = None):
        if not records:
            return
        path = self._path(table)
        if keys and path.exists():
            old = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            import pandas as pd
            merged = pd.DataFrame(old + list(records))
            merged = merged.drop_duplicates(subset=keys, keep="last")
            path.write_text("\n".join(json.dumps(rec, ensure_ascii=False) for rec in merged.to_dict(orient="records")) + "\n", encoding="utf-8")
        else:
            with path.open("a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

class CSVStore:
    def __init__(self, base: Path):
        self.base = Path(base); self.base.mkdir(parents=True, exist_ok=True)
    def _path(self, table: str) -> Path:
        return self.base / f"{table}.csv"
    def upsert(self, table: str, records: List[Mapping[str, Any]], keys: Optional[List[str]] = None):
        if not records: return
        path = self._path(table)
        df = pd.DataFrame(records)
        if path.exists():
            old = pd.read_csv(path)
            merged = pd.concat([old, df], ignore_index=True)
            if keys and all(k in merged.columns for k in keys):
                merged = merged.drop_duplicates(subset=keys, keep="last")
            merged.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)

class ParquetStore:
    def __init__(self, base: Path):
        if not _HAS_PARQUET:
            raise RuntimeError("pyarrow not installed. Install with 'pip install elexon-dl[parquet]'")
        self.base = Path(base); self.base.mkdir(parents=True, exist_ok=True)
    def _path(self, table: str) -> Path:
        return self.base / f"{table}.parquet"
    def upsert(self, table: str, records: List[Mapping[str, Any]], keys: Optional[List[str]] = None):
        if not records: return
        import pyarrow as pa, pyarrow.parquet as pq
        path = self._path(table)
        batch = pa.Table.from_pylist(list(records))
        if path.exists() and keys:
            existing = pq.read_table(path)
            if all(k in existing.column_names and k in batch.column_names for k in keys):
                merged = pa.concat_tables([existing, batch]).drop_duplicates(keys=keys, keep="last")
                pq.write_table(merged, path); return
        pq.write_table(batch, path)
