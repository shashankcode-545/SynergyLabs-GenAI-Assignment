"""LanceDB access: one table per (embedding model, dim, chunk config)."""
import re

import lancedb
import pyarrow as pa

FILTER_FIELDS = {"source_type", "doc_id"}


def schema(dim):
    return pa.schema([
        pa.field("chunk_id", pa.string()), pa.field("doc_id", pa.string()), pa.field("source_type", pa.string()),
        pa.field("title", pa.string()), pa.field("location", pa.string()), pa.field("text", pa.string()),
        pa.field("char_start", pa.int32()), pa.field("char_end", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), dim))])


def open_table(db_path, name, dim):
    return lancedb.connect(db_path).create_table(name, schema=schema(dim), exist_ok=True)


def where_clause(filters):
    """Build a SQL filter only from allow-listed fields and safe literal values (no injection through `filters`)."""
    parts = []
    for k, v in (filters or {}).items():
        if k not in FILTER_FIELDS or not re.fullmatch(r"[A-Za-z0-9._-]+", str(v)):
            raise ValueError(f"bad filter {k}={v!r}")
        parts.append(f"{k} = '{v}'")
    return " AND ".join(parts) or None


def search(table, qvec, k, filters=None):
    q = table.search(qvec.tolist()).distance_type("cosine").limit(k)
    w = where_clause(filters)
    if w:
        q = q.where(w, prefilter=True)
    rows = q.to_list()
    for r in rows:
        r["score"] = 1.0 - r.pop("_distance")  # cosine similarity
        r.pop("vector", None)
    return rows
