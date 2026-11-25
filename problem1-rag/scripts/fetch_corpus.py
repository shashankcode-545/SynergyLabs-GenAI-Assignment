"""Download the 8-document corpus at pinned versions and record metadata.

Run from problem1-rag/:  ..\\.venv\\Scripts\\python scripts\\fetch_corpus.py
Idempotent: a file whose sha256 already matches data/corpus/sources.json is not downloaded again.
Writes data/corpus/{pdf,md,html}/<doc_id>.<ext>, data/corpus/sources.json and data/corpus/SOURCES.md.
"""
import hashlib, json, time
from pathlib import Path

import httpx
from pypdf import PdfReader

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
UA = "SynergyLabsAssignmentCorpusFetcher/1.0 (educational RAG evaluation project)"
OWASP_SHA = "eb87ba80e0698de911dcdb8af60830e97723e3fc"  # latest commit touching 2_0_vulns as of 2026-11-25
OWASP_RAW = f"https://raw.githubusercontent.com/OWASP/www-project-top-10-for-large-language-model-applications/{OWASP_SHA}/2_0_vulns/"
NIST_LIC = "U.S. Government work, not subject to copyright in the United States (17 U.S.C. 105)"
OWASP_LIC = "CC BY-SA 4.0 (repository LICENSE.md)"
WIKI_LIC = "CC BY-SA 4.0 (Wikipedia text)"

SOURCES = [  # doc_id, format, title, publisher, download_url, citation_url, license, version
    ("nist-ai-100-1", "pdf", "Artificial Intelligence Risk Management Framework (AI RMF 1.0), NIST AI 100-1", "NIST",
     "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf", "https://doi.org/10.6028/NIST.AI.100-1", NIST_LIC, "AI 100-1"),
    ("nist-ai-600-1", "pdf", "AI RMF: Generative Artificial Intelligence Profile, NIST AI 600-1", "NIST",
     "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf", "https://doi.org/10.6028/NIST.AI.600-1", NIST_LIC, "AI 600-1"),
    *[(f"owasp-{slug}", "md", f"OWASP Top 10 for LLM Applications 2025 - {title}", "OWASP GenAI Security Project",
       OWASP_RAW + fname, OWASP_RAW.replace("raw.githubusercontent.com", "github.com").replace(OWASP_SHA, "blob/" + OWASP_SHA) + fname,
       OWASP_LIC, f"commit {OWASP_SHA[:7]}")
      for slug, title, fname in [
          ("llm01-prompt-injection", "LLM01:2025 Prompt Injection", "LLM01_PromptInjection.md"),
          ("llm02-sensitive-information-disclosure", "LLM02:2025 Sensitive Information Disclosure", "LLM02_SensitiveInformationDisclosure.md"),
          ("llm06-excessive-agency", "LLM06:2025 Excessive Agency", "LLM06_ExcessiveAgency.md"),
          ("llm08-vector-and-embedding-weaknesses", "LLM08:2025 Vector and Embedding Weaknesses", "LLM08_VectorAndEmbeddingWeaknesses.md")]],
    *[(f"owasp-{slug}", "html", f"OWASP Top 10 for LLM Applications 2025 - {title} (web page)", "OWASP GenAI Security Project",
       url, url, "CC BY-SA 4.0 (OWASP GenAI Security Project)", "web page as retrieved")
      for slug, title, url in [
          ("llm07-system-prompt-leakage", "LLM07:2025 System Prompt Leakage", "https://genai.owasp.org/llmrisk/llm072025-system-prompt-leakage/"),
          ("llm09-misinformation", "LLM09:2025 Misinformation", "https://genai.owasp.org/llmrisk/llm092025-misinformation/")]],
]
MAGIC = {"pdf": lambda b: b.startswith(b"%PDF-"), "html": lambda b: b"<html" in b[:2000].lower(),
         "md": lambda b: b.lstrip().startswith(b"#")}


def main():
    manifest = CORPUS / "sources.json"
    known = {d["doc_id"]: d for d in json.loads(manifest.read_text(encoding="utf-8"))} if manifest.exists() else {}
    records, failed = [], []
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=60) as http:
        for doc_id, fmt, title, publisher, url, cite, lic, version in SOURCES:
            path = CORPUS / fmt / f"{doc_id}.{fmt}"
            old = known.get(doc_id)
            if old and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == old["sha256"]:
                records.append(old)
                print(f"skip  {doc_id} (sha256 matches)")
                continue
            r = http.get(url)
            if r.is_error:
                failed.append(f"{doc_id}: HTTP {r.status_code} {r.text[:160]!r}")
                continue
            content, how = r.content, "downloaded by scripts/fetch_corpus.py"
            if not MAGIC[fmt](content):  # downloaded bytes are untrusted: refuse anything that is not the expected type
                failed.append(f"{doc_id}: content is not {fmt}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            rec = {"doc_id": doc_id, "file": path.relative_to(CORPUS).as_posix(), "source_type": fmt, "title": title,
                   "publisher": publisher, "version": version, "download_url": url, "citation_url": cite, "license": lic,
                   "retrieved_at": time.strftime("%Y-%m-%d"), "retrieval": how, "bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest()}
            if fmt == "pdf":
                rec["pages"] = len(PdfReader(path).pages)
            records.append(rec)
            print(f"fetch {doc_id}: {rec['bytes']:,} bytes")
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    rows = "\n".join(f"| `{d['doc_id']}` | {d['source_type'].upper()} | {d['title']} | {d['version']} | "
                     f"[link]({d['citation_url']}) | {d['license']} | {d['retrieved_at']} |" for d in records)
    (CORPUS / "SOURCES.md").write_text(
        "# Corpus sources\n\nGenerated by `scripts/fetch_corpus.py` from `sources.json` (which also holds sha256, bytes and download URLs).\n\n"
        "| doc_id | Format | Title | Version | Source | License | Retrieved |\n|---|---|---|---|---|---|---|\n" + rows + "\n",
        encoding="utf-8")
    print(f"{len(records)} docs, {sum(d['bytes'] for d in records):,} bytes total")
    if failed:
        raise SystemExit("FAILED downloads (not recorded):\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
