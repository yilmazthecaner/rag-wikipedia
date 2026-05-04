from src.chunker import chunk_text


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_splits_and_sizes():
    text = ("This is a sentence. " * 50).strip()
    chunks = chunk_text(text, chunk_size=200, overlap=20, min_chunk=50)
    assert isinstance(chunks, list)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and len(c) <= 200 for c in chunks)
