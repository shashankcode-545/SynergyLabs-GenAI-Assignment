"""Idempotent ingestion: unchanged documents are skipped; changed ones are replaced; chunk_id upserts block duplicates."""
import hashlib, json

from .chunking import chunk_spans
from .loaders import load_corpus
from .store import open_table


def chunk_rows(doc, size, overlap):
    cfg = f"{size}/{overlap}"
    for start, end in chunk_spans(doc.text, size, overlap):
        text = doc.text[start:end]
        yield {"chunk_id": hashlib.sha256(f"{doc.doc_id}|{cfg}|{start}|{hashlib.sha256(text.encode()).hexdigest()}".encode()).hexdigest()[:32],
            "doc_id": doc.doc_id, "source_type": doc.source_type, "title": doc.title,
            "location": doc.location_at(start) or "", "text": text, "char_start": start, "char_end": end}


def ingest(corpus_dir, settings, llm, size=None, overlap=None, offline=False):
    size, overlap = size or settings.chunk_size, overlap or settings.chunk_overlap
    name = settings.table_name(size, overlap)
    table = open_table(settings.db_path, name, settings.embed_dim)
    manifest_path = settings.db_path / f"{name}.manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    before = table.count_rows()
    stats = {"table": name, "chunk_size": size, "chunk_overlap": overlap, "embed_model": settings.embed_model,
            "embed_dim": settings.embed_dim, "vectors_before": before, "skipped": [], "ingested": []}
    for doc in load_corpus(corpus_dir):
        h = hashlib.sha256(doc.text.encode()).hexdigest()
        if manifest.get(doc.doc_id) == h and table.count_rows(f"doc_id = '{doc.doc_id}'") > 0:
            stats["skipped"].append(doc.doc_id)
            continue
        rows = list(chunk_rows(doc, size, overlap))
        vecs = llm.embed([r["text"] for r in rows], settings.embed_model, settings.embed_dim, "RETRIEVAL_DOCUMENT", offline=offline)
        for r, v in zip(rows, vecs):
            r["vector"] = v.tolist()
        table.delete(f"doc_id = '{doc.doc_id}'")  # changed doc: drop its old chunks first
        table.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all().execute(rows)
        manifest[doc.doc_id] = h  # written only after the store write succeeded, so a crash self-heals next run
        manifest_path.write_text(json.dumps(manifest, indent=1))
        stats["ingested"].append(f"{doc.doc_id} ({len(rows)} chunks)")
    stats["vectors_after"] = table.count_rows()
    return stats
