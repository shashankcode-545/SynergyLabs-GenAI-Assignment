"""Phase 0 checks (offline): pinned model config is consistent, smoke evidence is clean, secrets stay out of git.

Run: .venv\\Scripts\\python -m pytest tests -q
"""
import json, re, subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
JUDGE, GEN, GEN2 = "gemini-3.6-flash", "gemini-3.1-flash-lite", "gemma-4-31b-it"
EMBED = "gemini-embedding-2"  # switched from gemini-embedding-001 (free-tier quota exhausted; user-approved)
KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_-]{20,}|AQ\.[0-9A-Za-z_-]{20,}")


def env_example(path):
    return dict(l.split("=", 1) for l in path.read_text(encoding="utf-8").splitlines() if "=" in l and not l.startswith("#"))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def test_p1_models_pinned():
    env = env_example(ROOT / "problem1-rag/.env.example")
    assert (env["RAG_JUDGE_MODEL"], env["RAG_GEN_MODEL"], env["RAG_EMBED_MODEL"], env["RAG_EMBED_DIM"]) == (JUDGE, GEN, EMBED, "768")
    assert float(env["BUDGET_USD"]) + float(env_example(ROOT / "problem2-llm-judge/.env.example")["BUDGET_USD"]) <= 5


def test_p2_judge_and_generator_pinned_and_independent():
    cfg = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8")) for p in (ROOT / "problem2-llm-judge/configs").glob("*.yaml")}
    assert {cfg["judge_v0"]["model"], cfg["judge_v1"]["model"]} == {JUDGE}
    assert {cfg["gen_v1"]["model"], cfg["gen_v2"]["model"]} == {GEN2}
    # self-enhancement guard precondition: judge family differs from generator family
    assert cfg["judge_v1"]["family"] != cfg["gen_v1"]["family"]


def test_old_judge_absent_outside_history():
    # gemini-3.8-flash may appear only in archived evidence, docs explaining the switch, and the smoke run's
    # model-catalog listing (its checks are asserted to use the pinned models in the test below)
    allowed = {"evidence/phase0_history", "evidence/phase0_summary.md", "reports/architecture_plan.md",
               "tests/test_phase0.py", "evidence/phase0_smoke.json"}
    hits = [f for f in git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
            if "3.8-flash" in (ROOT / f).read_text(encoding="utf-8", errors="ignore") and not any(f.startswith(a) for a in allowed)]
    assert hits == []


def test_smoke_evidence_all_ok_with_pinned_models():
    smoke = json.loads((ROOT / "evidence/phase0_smoke.json").read_text(encoding="utf-8"))
    checks = smoke["checks"]
    assert all(c["ok"] for c in checks.values()), {k: c for k, c in checks.items() if not c["ok"]}
    assert {checks[f"model:{r}"]["id"] for r in ("gen", "judge", "gen2")} == {GEN, JUDGE, GEN2}  # embed model re-verified in P1 runs
    assert checks["generate:judge"]["model_version"] == JUDGE
    assert all(abs(c - 1) < 1e-3 for c in checks["embed"]["cos_trunc_vs_api768"])  # MRL truncation == API 768-d


def test_env_ignored_and_untracked():
    assert ".env" not in git("ls-files").splitlines()
    assert subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT).returncode == 0


def test_no_key_like_strings_in_repo_files():
    files = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    assert [f for f in files if KEY_PATTERN.search((ROOT / f).read_text(encoding="utf-8", errors="ignore"))] == []
