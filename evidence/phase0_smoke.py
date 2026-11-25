"""Phase 0 smoke test: verify Gemini access, model availability, embedding behaviour, JSON output and usage fields.

Run:  .venv\\Scripts\\python evidence\\phase0_smoke.py
Writes evidence/phase0_smoke.json. Makes ~5-6 small API calls (< $0.001; uncached by design, it is a live check).
Never prints the API key. Superseded runs with the previous judge model are in evidence/phase0_history/.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(encoding="utf-8-sig")  # utf-8-sig: PowerShell 5.1 writes a BOM that would corrupt the first key name
if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    sys.exit("GEMINI_API_KEY is not set (env var or .env at repo root).")

# Must match problem1-rag/.env.example and problem2-llm-judge/configs/*.yaml (checked by tests/test_phase0.py)
REQUIRED = {"embed": "gemini-embedding-001", "gen": "gemini-3.1-flash-lite", "judge": "gemini-3.6-flash", "gen2": "gemma-4-31b-it"}
# retry transient 429/503 with exponential backoff (attempt 1 hit 503 "high demand"; see phase0_smoke_attempt1.json)
client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(
    attempts=5, initial_delay=2.0, max_delay=30.0, http_status_codes=[429, 500, 503])))
out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sdk": genai.__version__, "checks": {}}


def record(name, ok, **info):
    out["checks"][name] = {"ok": ok, **info}
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {json.dumps(info, default=str)[:300]}")


# 1. model availability
models = {m.name.removeprefix("models/"): list(m.supported_actions or []) for m in client.models.list()}
for role, mid in REQUIRED.items():
    record(f"model:{role}", mid in models, id=mid, actions=models.get(mid))
out["all_models"] = sorted(models)

# 2. embedding: dims, norms, MRL truncation == API 768
texts = ["The NIST AI RMF has four functions.", "Prompt injection manipulates model inputs.", "LanceDB stores vectors on disk."]
def embed(dim):
    r = client.models.embed_content(model=REQUIRED["embed"], contents=texts,
                                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=dim))
    return np.array([e.values for e in r.embeddings], dtype=np.float32)
e3072, e768 = embed(3072), embed(768)
trunc = e3072[:, :768] / np.linalg.norm(e3072[:, :768], axis=1, keepdims=True)
api768 = e768 / np.linalg.norm(e768, axis=1, keepdims=True)
record("embed", e3072.shape[1] == 3072 and e768.shape[1] == 768,
       dims=[e3072.shape[1], e768.shape[1]], norm3072=np.linalg.norm(e3072, axis=1).round(4).tolist(),
       norm768_raw=np.linalg.norm(e768, axis=1).round(4).tolist(),
       cos_trunc_vs_api768=(trunc * api768).sum(1).round(5).tolist())

# 3. JSON-schema generation + usage fields (generator and judge)
schema = {"type": "object", "properties": {"answer": {"type": "string"}, "score": {"type": "integer"}},
          "required": ["answer", "score"]}
for role, level in (("gen", "MINIMAL"), ("judge", "LOW")):
    cfg = types.GenerateContentConfig(temperature=0, seed=42, response_mime_type="application/json",
                                      response_json_schema=schema, max_output_tokens=256,
                                      thinking_config=types.ThinkingConfig(thinking_level=level))
    t = time.perf_counter()
    try:
        r = client.models.generate_content(model=REQUIRED[role], contents="Capital of France? Score your confidence 1-5.", config=cfg)
        u = r.usage_metadata
        record(f"generate:{role}", bool(json.loads(r.text)), thinking_level=level, text=r.text,
               latency_ms=round((time.perf_counter() - t) * 1000), model_version=r.model_version,
               usage={k: getattr(u, k) for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count")})
    except Exception as e:  # record and continue so every check is reported
        record(f"generate:{role}", False, error=f"{type(e).__name__}: {e}"[:400])

# 4. Gemma generator with the pinned output limit; gen configs use system_prompt, so test system_instruction first
gid = REQUIRED["gen2"]
for label, sys_prompt in (("system_instruction", "Answer in one sentence."), ("no_system_instruction", None)):
    t = time.perf_counter()
    try:
        r = client.models.generate_content(model=gid, contents="What is the capital of France?",
                                           config=types.GenerateContentConfig(temperature=0, seed=42, max_output_tokens=1024,
                                                                              system_instruction=sys_prompt))
        record(f"gemma:{label}", bool(r.text), id=gid, text=(r.text or "")[:120],
               latency_ms=round((time.perf_counter() - t) * 1000),
               usage={k: getattr(r.usage_metadata, k, None) for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")})
        if r.text:
            break  # system_instruction works, so the fallback form is not needed
    except Exception as e:
        record(f"gemma:{label}", False, id=gid, error=f"{type(e).__name__}: {e}"[:400])

Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print("wrote", Path(__file__).with_suffix(".json"))
