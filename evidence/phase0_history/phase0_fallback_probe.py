"""Phase 0 follow-up: probe judge fallbacks and Gemma's empty response (finish_reason).

Run after phase0_smoke.py showed gemini-3.8-flash 503 "high demand". ~8 tiny calls. Writes phase0_fallback_probe.json.
"""
import json, time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(encoding="utf-8-sig")
client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(
    attempts=3, initial_delay=2.0, max_delay=20.0, http_status_codes=[429, 500, 503])))
schema = {"type": "object", "properties": {"answer": {"type": "string"}, "score": {"type": "integer"}}, "required": ["answer", "score"]}
out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "probes": []}


def probe(model, cfg, prompt="Capital of France? Score your confidence 1-5."):
    t = time.perf_counter()
    try:
        r = client.models.generate_content(model=model, contents=prompt, config=cfg)
        c = r.candidates[0] if r.candidates else None
        u = r.usage_metadata
        rec = {"model": model, "ok": bool(r.text), "text": (r.text or "")[:120],
               "finish_reason": str(c.finish_reason) if c else None,
               "usage": {k: getattr(u, k, None) for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")}}
    except Exception as e:
        rec = {"model": model, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    rec["latency_ms"] = round((time.perf_counter() - t) * 1000)
    out["probes"].append(rec)
    print(json.dumps(rec))


judge_cfg = lambda: types.GenerateContentConfig(temperature=0, seed=42, response_mime_type="application/json",
                                                response_json_schema=schema, max_output_tokens=512,
                                                thinking_config=types.ThinkingConfig(thinking_level="LOW"))
for m in ("gemini-3.8-flash", "gemini-3.6-flash", "gemini-4-hydrogen", "gemini-3.5-flash-lite"):
    probe(m, judge_cfg())
for m in ("gemma-4-26b-a4b-it", "gemma-4-31b-it"):
    probe(m, types.GenerateContentConfig(temperature=0, max_output_tokens=512), "In one sentence: what is the capital of France?")

Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2), encoding="utf-8")
