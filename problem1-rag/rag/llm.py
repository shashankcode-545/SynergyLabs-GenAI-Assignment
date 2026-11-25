"""Gemini wrapper: response cache (= audit log), hard budget cap, RPM throttle, retries, usage + cost capture.

Cache files (committed, so every result can be replayed offline):
  cache/llm.jsonl      one line per generate call: key, model, prompt, raw text, usage, cost, latency
  cache/emb_<model>_<dim>.npz  embeddings keyed by sha256(model|dim|task|text)
Spend = sum of cost over cache entries; a call that could push it over BUDGET_USD is refused before it is made.
"""
import hashlib, json, time
from pathlib import Path

import numpy as np

PRICES = {  # USD per 1M tokens (input, output incl. thinking); aistudio.google.com/docs/pricing, read 2026-11-25
    "gemini-3.1-flash-lite": (0.25, 1.50), "gemini-3.6-flash": (0.75, 3.75), "gemini-embedding-001": (0.15, 0.0),
    "gemini-embedding-2": (0.20, 0.0)}
EMB2_PREFIX = {"RETRIEVAL_QUERY": "task: search result | query: ", "RETRIEVAL_DOCUMENT": "title: none | text: "}  # docs: embeddings page


class BudgetExceeded(RuntimeError):
    pass


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Gemini:
    def __init__(self, cache_dir, budget_usd, rpm=15, client=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = self.cache_dir / "llm.jsonl"
        self.entries = {}
        if self.log.exists():
            for line in self.log.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                self.entries[e["key"]] = e
        self.emb_spend = self._load_emb_spend()
        self.budget, self.min_interval, self._last = budget_usd, 60.0 / rpm, 0.0
        self._client = client
        self._emb = {}

    # -- plumbing --------------------------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(
                attempts=6, initial_delay=2.0, max_delay=60.0, http_status_codes=[429, 500, 503])))
        return self._client

    def spent(self):
        return sum(e["cost_usd"] for e in self.entries.values()) + self.emb_spend

    def _guard(self, est_cost):
        if self.spent() + est_cost > self.budget:
            raise BudgetExceeded(f"spent ${self.spent():.4f} + next ~${est_cost:.4f} > budget ${self.budget}")
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    @staticmethod
    def cost(model, prompt_tokens, output_tokens):
        p_in, p_out = PRICES.get(model, (0.0, 0.0))
        return (prompt_tokens * p_in + output_tokens * p_out) / 1e6

    # -- generation ------------------------------------------------------------------------------
    def generate(self, model, prompt, system=None, schema=None, thinking_level=None, max_output_tokens=1024,
                 temperature=0.0, seed=42, rerun_idx=0, offline=False):
        params = dict(system=system, schema=schema, thinking_level=thinking_level, max_output_tokens=max_output_tokens,
                      temperature=temperature, seed=seed, rerun_idx=rerun_idx)
        key = _key(model, prompt, params)
        if key in self.entries:
            return {**self.entries[key], "cached": True}
        if offline:
            raise KeyError(f"offline replay: no cached response for key {key[:12]}")
        from google.genai import types
        self._guard(self.cost(model, len(prompt) // 4 + 200, max_output_tokens))
        cfg = types.GenerateContentConfig(
            temperature=temperature, seed=seed, max_output_tokens=max_output_tokens, system_instruction=system,
            response_mime_type="application/json" if schema else None, response_json_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None)
        t = time.perf_counter()
        r = self.client.models.generate_content(model=model, contents=prompt, config=cfg)
        u = r.usage_metadata
        usage = {"prompt": u.prompt_token_count or 0, "output": u.candidates_token_count or 0,
                 "thoughts": u.thoughts_token_count or 0}
        e = {"key": key, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "model_version": r.model_version,
             "params": params, "prompt": prompt, "text": r.text or "", "usage": usage,
             "cost_usd": self.cost(model, usage["prompt"], usage["output"] + usage["thoughts"]),
             "latency_ms": round((time.perf_counter() - t) * 1000)}
        with self.log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.entries[key] = e
        return {**e, "cached": False}

    # -- embeddings ------------------------------------------------------------------------------
    def _emb_file(self, model, dim):
        return self.cache_dir / f"emb_{model}_{dim}.npz"

    def _load_emb(self, model, dim):
        if (model, dim) not in self._emb:
            f = self._emb_file(model, dim)
            self._emb[model, dim] = dict(zip(*(lambda z: (z["keys"].tolist(), list(z["vecs"])))(np.load(f)))) if f.exists() else {}
        return self._emb[model, dim]

    def _load_emb_spend(self):
        f = self.cache_dir / "emb_spend.json"
        return json.loads(f.read_text())["usd"] if f.exists() else 0.0

    def embed(self, texts, model, dim, task_type, offline=False, batch=100):
        """-> float32 [n, dim], L2-normalized (gemini-embedding-001 returns unnormalized vectors below 3072-d)."""
        cache = self._load_emb(model, dim)
        keys = [_key(model, dim, task_type, t) for t in texts]
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in cache]
        if todo and offline:
            raise KeyError(f"offline replay: {len(todo)} embeddings not cached")
        from google.genai import types
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            est = self.cost(model, sum(len(t) for _, t in chunk) // 4, 0)
            self._guard(est)
            if model in PRICES and model.startswith("gemini-embedding-2"):
                # embedding-2: no task_type (prompt prefix instead); a plain list of strings would be merged into ONE
                # embedding, so each text goes in its own Content
                contents = [types.Content(parts=[types.Part(text=EMB2_PREFIX[task_type] + t)]) for _, t in chunk]
                cfg = types.EmbedContentConfig(output_dimensionality=dim)
            else:
                contents = [t for _, t in chunk]
                cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)
            r = self.client.models.embed_content(model=model, contents=contents, config=cfg)
            if len(r.embeddings) != len(chunk):
                raise RuntimeError(f"expected {len(chunk)} embeddings, got {len(r.embeddings)}")
            for (k, _), e in zip(chunk, r.embeddings):
                v = np.asarray(e.values, dtype=np.float32)
                cache[k] = v / np.linalg.norm(v)
            self.emb_spend += est  # shortcut: embed API returns no token usage, so cost uses chars/4; fine at $0.15/1M
            (self.cache_dir / "emb_spend.json").write_text(json.dumps({"usd": self.emb_spend}))
            np.savez(self._emb_file(model, dim), keys=np.array(list(cache)), vecs=np.stack(list(cache.values())))
        return np.stack([cache[k] for k in keys]).astype(np.float32)
