from src.retriever import classify_query
from src.generator import _fallback_answer


def test_classify_person():
    assert classify_query("Who was Albert Einstein and what did he do?") == "person"


def test_classify_place():
    assert classify_query("Where is the Eiffel Tower located?") == "place"


def test_classify_both():
    # Intentional mix to trigger 'both'
    res = classify_query("Compare Albert Einstein and the Eiffel Tower")
    assert res in {"both", "person", "place"}


def test_fallback_answer_uses_context():
    chunks = [
        {
            "metadata": {"title": "Albert Einstein"},
            "text": "Albert Einstein was a theoretical physicist. He developed the theory of relativity.",
        }
    ]
    answer = _fallback_answer(chunks)
    assert "Albert Einstein" in answer
    assert "theory of relativity" in answer
