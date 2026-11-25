"""Settings from environment (repo-root .env). Defaults = the values stated in .env.example / README."""
import json, os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT.parent / ".env", encoding="utf-8-sig")


def _calibrated_tau():
    f = ROOT / "eval" / "calibration.json"
    return json.loads(f.read_text())["min_similarity"] if f.exists() else 0.5


def _env(name, default):
    v = os.getenv(name)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    embed_model: str = _env("RAG_EMBED_MODEL", "gemini-embedding-2")
    embed_dim: int = int(_env("RAG_EMBED_DIM", 768))
    gen_model: str = _env("RAG_GEN_MODEL", "gemini-3.1-flash-lite")
    judge_model: str = _env("RAG_JUDGE_MODEL", "gemini-3.6-flash")
    chunk_size: int = int(_env("RAG_CHUNK_SIZE", 1200))
    chunk_overlap: int = int(_env("RAG_CHUNK_OVERLAP", 200))
    top_k: int = int(_env("RAG_TOP_K", 5))
    min_similarity: float = float(_env("RAG_MIN_SIMILARITY", _calibrated_tau()))
    db_path: Path = ROOT / _env("RAG_DB_PATH", "lancedb")
    log_path: Path = ROOT / _env("RAG_LOG_PATH", "logs/queries.jsonl")
    cache_dir: Path = ROOT / "cache"
    budget_usd: float = float(_env("BUDGET_USD", 1.5))
    rpm: int = int(_env("LLM_RPM", 15))

    def table_name(self, size=None, overlap=None):
        return f"chunks_{self.embed_model.replace('-', '_')}_{self.embed_dim}_c{size or self.chunk_size}_o{overlap or self.chunk_overlap}"
