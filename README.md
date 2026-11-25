# Synergy Labs — Applied AI/ML Engineering Take-Home

Two independent Python 3.11 applications (Windows-tested), sharing no code:

| Folder | Problem | README |
|---|---|---|
| [`problem1-rag/`](problem1-rag/) | Cost-efficient RAG application (LanceDB + Gemini) | [problem1-rag/README.md](problem1-rag/README.md) |
| [`problem2-llm-judge/`](problem2-llm-judge/) | LLM-as-Judge evaluation pipeline with bias measurement | (built in Phase 3) |

Other folders:
- `reports/`: [architecture plan](reports/architecture_plan.md), and later the filled submission template.
- `evidence/`: Phase 0 model checks ([summary](evidence/phase0_summary.md)), and later the screenshot checklist.
- `reference/`: the assignment brief and submission template, as given.

## Setup

```powershell
py -3.11 -m venv .venv                  # or reuse the existing .venv
.venv\Scripts\python -m pip install -r requirements.lock.txt
Copy-Item .env.example .env             # then put your GEMINI_API_KEY in .env (never committed)
```

**Number policy.** Every reported number is tagged:
- **[MEASURED]**: from our own runs and logs;
- **[PROJECTED]**: public prices × stated assumptions;
- **[ESTIMATE]**: planning only.

Every API call is cached and replayable. A hard `BUDGET_USD=5` cap applies, and billing is never enabled by the code.

## Progress

| Milestone | Status | Commit | Tests |
|---|---|---|---|
| Phase 0: scaffold, pinned models, Gemini smoke test | done | `cd907dc` | `tests/` 6 passed |
| Phase 1a: P1 corpus (8 docs), 18 eval + 5 dev questions, validators | done, **pending candidate review** of the questions | this milestone | `problem1-rag/tests/` |
| Phase 1b: ingestion (chunking, embeddings, LanceDB, idempotency) | not started | | |
| Phase 1c: query path + HTTP API + logging | not started | | |
| Phase 2: P1 evaluation (retrieval, answer, latency, cost) | not started | | |
| Phases 3–4: P2 judge pipeline, bias + validation runs | not started | | |
| Phase 5: packaging, submission template | not started | | |

**Pinned models (Phase 0):**
- embeddings: `gemini-embedding-001` @ 768-d
- RAG generator: `gemini-3.1-flash-lite`
- only judge: `gemini-3.6-flash`
- second-family generator: `gemma-4-31b-it`

## Tests

```powershell
.venv\Scripts\python -m pytest tests -q                              # Phase 0 repo checks
cd problem1-rag; ..\.venv\Scripts\python -m pytest -q; cd ..          # Problem 1
```
