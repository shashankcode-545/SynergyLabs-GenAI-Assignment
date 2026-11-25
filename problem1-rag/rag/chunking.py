"""Recursive character chunking with exact char offsets into the document text."""

SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_spans(text, size, overlap):
    """-> [(start, end)] covering `text`; each span <= size chars, consecutive spans overlap ~`overlap` chars."""
    assert 0 <= overlap < size
    spans, start, n = [], 0, len(text)
    while start < n:
        while start < n and text[start].isspace():
            start += 1
        if start >= n:
            break
        end = min(start + size, n)
        if end < n:
            for sep in SEPARATORS:
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        stripped_end = end
        while stripped_end > start and text[stripped_end - 1].isspace():
            stripped_end -= 1
        spans.append((start, stripped_end))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt, end)
        start = space + 1 if space != -1 else nxt
    return spans
