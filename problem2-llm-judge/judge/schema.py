"""Test-case and verdict schemas; suite loading (JSON / JSONL / YAML)."""
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

RUBRIC = yaml.safe_load((Path(__file__).with_name("rubric.yaml")).read_text(encoding="utf-8"))
CRITERIA = list(RUBRIC["criteria"])


class TestCase(BaseModel):
    __test__ = False
    id: str
    input: str
    system_prompt: str = ""
    model_output: str
    expected_output: str | None = None
    criteria: list[str] | None = None
    meta: dict = {}


class Criterion(BaseModel):
    score: int = Field(ge=1, le=5)
    evidence: str = ""
    rationale: str


class PointwiseV1(BaseModel):
    criteria: dict[str, Criterion]
    overall_rationale: str


class PointwiseV0(BaseModel):
    score: int = Field(ge=1, le=10)
    rationale: str


class PairwiseV1(BaseModel):
    per_criterion: dict[str, Literal["A", "B", "tie"]]
    winner: Literal["A", "B", "tie"]
    decisive_quote: str = ""
    rationale: str


class PairwiseV0(BaseModel):
    winner: Literal["A", "B", "tie"]
    rationale: str


def load_suite(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif path.suffix == ".jsonl":
        data = [json.loads(l) for l in text.splitlines() if l.strip()]
    else:
        data = json.loads(text)
    cases = data["cases"] if isinstance(data, dict) else data
    return [TestCase.model_validate(c) for c in cases]


def save_suite(cases, path):
    Path(path).write_text("\n".join(json.dumps(c.model_dump(), ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
