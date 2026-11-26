"""PDF / HTML / Markdown -> normalized text + location map (page or section) per document.

The same text is used for chunking (char offsets) and for resolving gold evidence quotes, so labels and chunks
share one coordinate system.
"""
import json, re, unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

BLOCKS = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "blockquote", "pre", "figcaption", "td", "th"]
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
HTML_DROP = ["script", "style", "sup.reference", "sup.mw-ref", ".mw-references-wrap", "ol.references", ".reflist",
                ".navbox", ".mw-editsection", ".hatnote", ".metadata", ".noprint", "link", "meta"]


@dataclass
class Document:
    doc_id: str
    source_type: str
    title: str
    text: str
    locations: list = field(default_factory=list)

    def location_at(self, offset):
        label = None
        for start, lab in self.locations:
            if start > offset:
                break
            label = lab
        return label


def normalize(s):
    s = unicodedata.normalize("NFKC", s).replace("­", "")
    lines = [re.sub(r"[ \t ]+", " ", l).strip() for l in s.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _join(parts):
    """parts: [(label, text)] -> (text, locations) with blocks separated by blank lines."""
    text, locs = "", []
    for label, t in parts:
        t = normalize(t)
        if not t:
            continue
        if text:
            text += "\n\n"
        if label is not None and (not locs or locs[-1][1] != label):
            locs.append((len(text), label))
        text += t
    return text, locs


def _pdf(path):
    return _join((f"p. {i}", page.extract_text() or "") for i, page in enumerate(PdfReader(path).pages, 1))


def _html(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for sel in HTML_DROP:
        for el in soup.select(sel):
            el.decompose()
    parts, section = [], None
    root = soup.find("main") or soup.find("article") or soup
    for el in root.find_all(BLOCKS):
        if el.find(BLOCKS):
            continue
        t = el.get_text()
        if el.name in HEADINGS:
            section = normalize(t)
        parts.append((section, t))
    return _join(parts)


def _md(path):
    parts, section, buf = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", line)
        if line.startswith("#"):
            parts.append((section, "\n".join(buf)))
            section, buf = line.lstrip("#").strip(), []
        buf.append(line)
    parts.append((section, "\n".join(buf)))
    return _join(parts)


LOADERS = {"pdf": _pdf, "html": _html, "md": _md}


def load_document(path, doc_id, title, source_type=None):
    path = Path(path)
    source_type = source_type or path.suffix.lstrip(".").lower()
    text, locations = LOADERS[source_type](path)
    return Document(doc_id, source_type, title, text, locations)


def load_corpus(corpus_dir):
    corpus_dir = Path(corpus_dir)
    meta = json.loads((corpus_dir / "sources.json").read_text(encoding="utf-8"))
    return [load_document(corpus_dir / m["file"], m["doc_id"], m["title"], m["source_type"]) for m in meta]


def find_span(text, quote):
    """Whitespace-insensitive exact match of `quote` in `text` -> (start, end) offsets, or None."""
    words = normalize(quote).split()
    m = re.search(r"\s+".join(map(re.escape, words)), text) if words else None
    return (m.start(), m.end()) if m else None
