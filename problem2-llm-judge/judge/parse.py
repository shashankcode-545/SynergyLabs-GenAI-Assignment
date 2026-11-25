"""Structured-verdict parsing with a malformed-JSON fallback chain:
1 json.loads -> 2 strip ``` fences -> 3 extract outermost {...} -> 4 json_repair -> 5 pydantic schema validation.
Returns (verdict | None, step_that_worked | "parse_failed", error).
"""
import json, re

from json_repair import repair_json
from pydantic import ValidationError


def _candidates(text):
    yield "json", text
    fenced = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    yield "fences", fenced
    a, b = fenced.find("{"), fenced.rfind("}")
    if a != -1 and b > a:
        yield "braces", fenced[a:b + 1]
    yield "repair", repair_json(fenced[a:] if a != -1 else fenced)


def parse_verdict(text, model):
    err = "empty response"
    for step, cand in _candidates(text or ""):
        try:
            obj = json.loads(cand) if isinstance(cand, str) else cand
            return model.model_validate(obj), step, None
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
            err = f"{step}: {str(e).splitlines()[0][:200]}"
    return None, "parse_failed", err
