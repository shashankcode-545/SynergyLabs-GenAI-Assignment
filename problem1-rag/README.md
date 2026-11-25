# Problem 1 — Cost-Efficient RAG Application

A QA service over a small, mixed-format document corpus, backed by an embedded **LanceDB** vector store. It is evaluated honestly on retrieval quality, answer quality, latency and cost. Design: [`../reports/architecture_plan.md`](../reports/architecture_plan.md) (§3, §5.1).

> **Status:** Phase 1a done: corpus + frozen question sets + validators + tests.
> Not built yet: ingestion (1b), query API (1c), evaluation (2).

## Setup (Windows, PowerShell)

```powershell
cd problem1-rag
..\.venv\Scripts\python -m pip install -r requirements.txt
```

The Python 3.11 venv lives at the repo root (`.venv`). Copy `.env.example` → `../.env` and set `GEMINI_API_KEY`. The key is not needed for anything in Phase 1a.

## Corpus (`data/corpus/`)

There are 8 public documents in 3 formats, each pinned to an exact version. Full metadata (URL, license, version, sha256, bytes, pages, retrieval method) is in [`data/corpus/sources.json`](data/corpus/sources.json); a readable table is in [`data/corpus/SOURCES.md`](data/corpus/SOURCES.md).

| Format | Documents | Pinned by |
|---|---|---|
| PDF (2) | NIST AI 100-1 (AI RMF 1.0), NIST AI 600-1 (Generative AI Profile) | URL + sha256 |
| Markdown (4) | OWASP Top 10 for LLM Apps 2025: LLM01, LLM02, LLM06, LLM08 | git commit `eb87ba8` + sha256 |
| HTML (2) | OWASP Top 10 for LLM Apps 2025 web pages: LLM07 System Prompt Leakage, LLM09 Misinformation | sha256 (retrieved 2026-11-25) |

```powershell
..\.venv\Scripts\python scripts\fetch_corpus.py   # re-fetch / verify; skips files whose sha256 already matches
```

Wikipedia was dropped: its robot policy blocks scripted downloads (HTTP 403), so two OWASP web pages replaced it.

**Text extraction** (`rag/loaders.py`) produces one normalized text per document plus a page map (PDF) or a section map (MD/HTML). Chunk offsets and gold-evidence offsets both use this same text.

## Evaluation questions (`eval/`)

| File | n | Composition | Use |
|---|---|---|---|
| `questions.jsonl` | **18** | 6 single-fact · 3 paraphrased · 3 multi-span · 2 filter-scoped · 4 unanswerable (14 answerable) | **all reported metrics** |
| `dev_questions.jsonl` | 5 | 3 answerable · 2 unanswerable | tuning only (similarity threshold, prompt debugging). **Never reported** |

Each answerable question has a short `gold_answer` (for EM/F1) and one or more `evidence` quotes `{doc_id, quote}`. A retrieved chunk counts as relevant when it covers ≥ 50% of an evidence span (span overlap in the shared text coordinates). Relevance therefore doesn't depend on chunk size.

`eval/validate_questions.py` enforces the ground truth mechanically:

| Check | Why it matters |
|---|---|
| Every quote is found verbatim (whitespace-insensitive) in its document's extracted text | evidence can't be invented or paraphrased |
| The gold answer (or each `gold_answer_part`) is contained in the evidence quotes | answers are extractive and verifiable |
| `multi_span` has ≥ 2 non-overlapping spans | the multi-span label is true |
| `paraphrased` shares ≤ 25% of its content words with its evidence | guards against easy wins from shared wording |
| `filter_scoped`: evidence matches the filter, and `topic_term` also appears in a document the filter excludes | the metadata filter actually matters |
| `unanswerable`: none of its `absent_terms` appears anywhere in the corpus | the "no answer in corpus" label holds up |
| dev and eval share no ids, no question text and **no overlapping evidence spans** | tuning can't leak into reported results |

```powershell
..\.venv\Scripts\python -m eval.validate_questions   # prints per-question type / overlap, exits 1 on any failure
```

The questions were drafted by Claude (AI-assisted) from the extracted text and are pending candidate review. Edits are cheap: re-run the validator and the tests.

## Tests

```powershell
..\.venv\Scripts\python -m pytest -q   # offline; no API key needed
```

- `test_loaders.py`: normalization, whitespace-insensitive quote matching, Markdown sections and link stripping, HTML citation/script removal.
- `test_corpus.py`: 8 documents with format mix 2/4/2, complete metadata, sha256 and byte-size match, under 10 MB in total, every document extracts ≥ 3,000 chars with locations.
- `test_questions.py`: the frozen sets pass every validator check, the sizes match the plan, and the validator **rejects** invented quotes, non-extractive gold answers, an unanswerable question whose answer is in the corpus, a high-overlap "paraphrase", a moot filter, and dev/eval evidence leaks.

## Licenses

NIST publications are U.S. Government works (17 U.S.C. 105). OWASP texts are CC BY-SA 4.0, attributed in `SOURCES.md`; the files are unmodified.
