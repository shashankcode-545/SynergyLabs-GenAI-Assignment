"""HTTP API: POST /query, GET /stats, GET /health.  Run: python -m rag serve"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answer import answer
from .config import Settings
from .llm import BudgetExceeded, Gemini
from .store import open_table

settings = Settings()
llm = Gemini(settings.cache_dir, settings.budget_usd, settings.rpm)
app = FastAPI(title="Cost-efficient RAG (LanceDB + Gemini)")


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=settings.top_k, ge=1, le=20)
    filters: dict[str, str] = {}


def table():
    return open_table(settings.db_path, settings.table_name(), settings.embed_dim)


@app.post("/query")
def query(q: QueryIn):
    try:
        return answer(q.question, settings, llm, table(), k=q.k, filters=q.filters)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except BudgetExceeded as e:
        raise HTTPException(429, str(e))


@app.get("/stats")
def stats():
    return {"table": settings.table_name(), "vectors": table().count_rows(), "embed_model": settings.embed_model,
            "embed_dim": settings.embed_dim, "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
            "gen_model": settings.gen_model, "min_similarity": settings.min_similarity, "spent_usd": round(llm.spent(), 4)}


@app.get("/health")
def health():
    return {"ok": True}
