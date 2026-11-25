"""Loader unit tests on synthetic inputs (no corpus needed)."""
from rag.loaders import Document, find_span, load_document, normalize


def test_normalize_collapses_spaces_and_blank_lines_but_keeps_paragraphs():
    assert normalize("a  \t b c\n\n\n\nd­e") == "a b c\n\nde"  # soft hyphen removed
    assert normalize("ﬁle") == "file"  # NFKC ligature


def test_find_span_is_whitespace_insensitive_and_exact_otherwise():
    text = "The Core is composed of four functions: GOVERN, MAP,\nMEASURE, and MANAGE."
    start, end = find_span(text, "four functions: GOVERN, MAP, MEASURE")
    assert text[start:end] == "four functions: GOVERN, MAP,\nMEASURE"
    assert find_span(text, "four functions: govern") is None  # case-sensitive
    assert find_span(text, "") is None


def test_markdown_sections_and_link_stripping(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("# Title\nintro [link text](http://x.y) here\n## Part A\nalpha\n## Part B\nbeta\n", encoding="utf-8")
    d = load_document(p, "x", "X")
    assert "link text" in d.text and "http://x.y" not in d.text
    assert [lab for _, lab in d.locations] == ["Title", "Part A", "Part B"]
    assert d.location_at(d.text.index("beta")) == "Part B"


def test_html_drops_citations_and_tracks_sections(tmp_path):
    p = tmp_path / "x.html"
    p.write_text("<html><body><h2>Intro</h2><p>RAG combines <a href='#'>retrieval</a> and generation."
                 "<sup class='mw-ref reference'>[1]</sup></p><script>evil()</script>"
                 "<h2>History</h2><ul><li>first item</li></ul></body></html>", encoding="utf-8")
    d = load_document(p, "x", "X")
    assert "RAG combines retrieval and generation." in d.text  # inline link text joined, no newline split
    assert "[1]" not in d.text and "evil" not in d.text
    assert d.location_at(d.text.index("first item")) == "History"


def test_location_at_before_first_label():
    assert Document("d", "pdf", "t", "abc", [(1, "p. 1")]).location_at(0) is None
