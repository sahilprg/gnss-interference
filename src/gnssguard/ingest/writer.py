"""Buffered Parquet writer, one file per region and day."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gnssguard.ingest.trace import COLUMNS, SCHEMA


class RegionWriter:
    """Collect rows per region and flush them to zstd-compressed Parquet."""

    def __init__(self, out_dir: Path, day: str, flush_rows: int = 250_000):
        self.out_dir = Path(out_dir)
        self.day = day
        self.flush_rows = flush_rows
        self._buf: dict[str, list[tuple]] = {}
        self._writers: dict[str, pq.ParquetWriter] = {}
        self.rows: dict[str, int] = {}

    def path_for(self, region: str) -> Path:
        return self.out_dir / f"region={region}" / f"date={self.day}" / "part-0.parquet"

    def add(self, region: str, row: tuple) -> None:
        buf = self._buf.setdefault(region, [])
        buf.append(row)
        if len(buf) >= self.flush_rows:
            self._flush(region)

    def _flush(self, region: str) -> None:
        buf = self._buf.get(region)
        if not buf:
            return
        cols = list(zip(*buf, strict=True))
        table = pa.Table.from_arrays(
            [pa.array(c, type=f.type) for c, f in zip(cols, SCHEMA, strict=True)],
            schema=SCHEMA,
        )
        w = self._writers.get(region)
        if w is None:
            path = self.path_for(region)
            path.parent.mkdir(parents=True, exist_ok=True)
            w = pq.ParquetWriter(path, SCHEMA, compression="zstd", compression_level=6)
            self._writers[region] = w
        w.write_table(table)
        self.rows[region] = self.rows.get(region, 0) + len(buf)
        buf.clear()

    def close(self) -> dict[str, int]:
        for region in list(self._buf):
            self._flush(region)
        for w in self._writers.values():
            w.close()
        self._writers.clear()
        return dict(self.rows)


assert COLUMNS == SCHEMA.names
