"""Gemini wrapper for the judge app: response cache = audit log, hard budget cap, RPM throttle, retries, usage + cost.

cache/calls.jsonl holds one line per call: key, tag, model, full prompt + system, raw response text, usage, cost, latency.
It is committed, so every verdict can be replayed offline (`--offline`) and audited. Spend = sum of logged costs; a call
that could push spend over BUDGET_USD is refused before it is made.
"""
import hashlib, json, os, time
from pathlib import Path

PRICES = {"gemini-3.6-flash": (0.75, 3.75), "gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.1-flash-lite": (0.25, 1.50), "gemma-4-31b-it": (0.0, 0.0)}


class BudgetExceeded(RuntimeError):
    pass


class Gemini:
    def __init__(self, cache_file, budget_usd, rpm=15, client=None, offline=False):
        self.path = Path(cache_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        for f in sorted(self.path.parent.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                self.entries[e["key"]] = e
        self.budget, self.min_interval, self._last = budget_usd, 60.0 / rpm, 0.0
        self._client, self.offline = client, offline
        self.calls = self.cached_hits = 0

    @property
    def client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            timeout_ms = int(float(os.getenv("LLM_TIMEOUT_S") or 120) * 1000)
            key = os.getenv("GEMINI_API_KEY_P2")
            if not key:
                raise RuntimeError("GEMINI_API_KEY_P2 is not set (repo-root .env)")
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout_ms, retry_options=types.HttpRetryOptions(
                attempts=int(os.getenv("LLM_RETRY_ATTEMPTS") or 6), initial_delay=2.0, max_delay=60.0, http_status_codes=[429, 500, 503, 504])))
        return self._client

    def spent(self):
        return sum(e["cost_usd"] for e in self.entries.values())

    @staticmethod
    def cost(model, prompt_tokens, output_tokens):
        p_in, p_out = PRICES.get(model, (0.0, 0.0))
        return (prompt_tokens * p_in + output_tokens * p_out) / 1e6

    def generate(self, model, prompt, system=None, schema=None, thinking_level=None, max_output_tokens=2048,
                 temperature=0.0, seed=42, rerun_idx=0, tag=""):
        params = dict(system=system, schema=schema, thinking_level=thinking_level, max_output_tokens=max_output_tokens,
                      temperature=temperature, seed=seed, rerun_idx=rerun_idx)
        key = hashlib.sha256(json.dumps([model, prompt, params], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in self.entries:
            self.cached_hits += 1
            return {**self.entries[key], "cached": True}
        if self.offline:
            raise KeyError(f"offline replay: no cached response for {tag} key {key[:12]}")
        est = self.cost(model, len(prompt) // 4 + 300, max_output_tokens)
        if self.spent() + est > self.budget:
            raise BudgetExceeded(f"spent ${self.spent():.4f} + next ~${est:.4f} > budget ${self.budget}")
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        from google.genai import types
        cfg = types.GenerateContentConfig(
            temperature=temperature, seed=seed, max_output_tokens=max_output_tokens, system_instruction=system,
            response_mime_type="application/json" if schema else None, response_json_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None)
        t = time.perf_counter()
        r = self.client.models.generate_content(model=model, contents=prompt, config=cfg)
        u = r.usage_metadata
        usage = {"prompt": u.prompt_token_count or 0, "output": u.candidates_token_count or 0,
                 "thoughts": u.thoughts_token_count or 0}
        e = {"key": key, "tag": tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model,
             "model_version": r.model_version, "params": params, "prompt": prompt, "text": r.text or "", "usage": usage,
             "cost_usd": self.cost(model, usage["prompt"], usage["output"] + usage["thoughts"]),
             "latency_ms": round((time.perf_counter() - t) * 1000)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.entries[key] = e
        self.calls += 1
        return {**e, "cached": False}
