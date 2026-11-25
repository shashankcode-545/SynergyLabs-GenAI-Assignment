"""Corpus integrity: the 8 pinned documents exist, match their recorded hashes and extract to usable text."""
import hashlib, json
from collections import Counter
from pathlib import Path

import pytest

from rag.loaders import load_corpus

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
META = json.loads((CORPUS / "sources.json").read_text(encoding="utf-8"))
FIELDS = ["doc_id", "file", "source_type", "title", "publisher", "version", "download_url", "citation_url", "license",
          "retrieved_at", "retrieval", "bytes", "sha256"]


def test_eight_docs_with_expected_format_mix():
    assert len(META) == 8 and len({m["doc_id"] for m in META}) == 8
    assert Counter(m["source_type"] for m in META) == {"pdf": 2, "md": 4, "html": 2}


@pytest.mark.parametrize("m", META, ids=lambda m: m["doc_id"])
def test_metadata_complete_and_file_matches_hash(m):
    assert all(m.get(f) not in (None, "") for f in FIELDS), [f for f in FIELDS if m.get(f) in (None, "")]
    data = (CORPUS / m["file"]).read_bytes()
    assert len(data) == m["bytes"] and hashlib.sha256(data).hexdigest() == m["sha256"]
    assert m["file"] == f"{m['source_type']}/{m['doc_id']}.{m['source_type']}"
    if m["source_type"] == "pdf":
        assert data.startswith(b"%PDF-") and m["pages"] > 0
    if m["source_type"] == "html":
        assert b"<html" in data[:2000].lower()


def test_total_size_under_10mb():
    assert sum(m["bytes"] for m in META) < 10 * 1024 * 1024


def test_every_doc_extracts_text_with_locations():
    docs = load_corpus(CORPUS)
    assert [d.doc_id for d in docs] == [m["doc_id"] for m in META]
    for d, m in zip(docs, META):
        assert len(d.text) >= 3000, d.doc_id
        assert d.locations, d.doc_id
        if m["source_type"] == "pdf":
            assert all(lab.startswith("p. ") for _, lab in d.locations)
