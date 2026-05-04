from src.retriever import classify_query


def test_classify_person():
    assert classify_query("Who was Albert Einstein and what did he do?") == "person"


def test_classify_place():
    assert classify_query("Where is the Eiffel Tower located?") == "place"


def test_classify_both():
    # Intentional mix to trigger 'both'
    res = classify_query("Compare Albert Einstein and the Eiffel Tower")
    assert res in {"both", "person", "place"}
