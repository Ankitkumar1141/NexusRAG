import pytest
from src.cache import get, put, clear, stats, _hash, _normalise

@pytest.fixture(autouse=True)
def clean_cache():
    clear()
    yield
    clear()

def test_cache_put_and_get():
    question = "What are the core working hours?"
    result = {"answer": "10:00 to 16:00", "sources": ["handbook.pdf"], "debug": {}}
    
    assert get(question) is None
    put(question, result)
    
    cached = get(question)
    assert cached is not None
    assert cached["answer"] == "10:00 to 16:00"

def test_cache_normalisation():
    q1 = "What are the core working hours?"
    q2 = "  what ARE the core working hours?  "
    result = {"answer": "10:00 to 16:00"}
    
    put(q1, result)
    cached = get(q2)
    assert cached is not None
    assert cached["answer"] == "10:00 to 16:00"

def test_cache_stats():
    q = "What is the parental leave policy?"
    res = {"answer": "16 weeks"}
    
    get(q)  # miss
    put(q, res)
    get(q)  # hit
    get(q)  # hit
    
    s = stats()
    assert s["size"] == 1
    assert s["hits"] == 2
    assert s["misses"] == 1

def test_cache_clear():
    put("Question 1", {"answer": "A1"})
    assert stats()["size"] == 1
    clear()
    assert stats()["size"] == 0
    assert stats()["hits"] == 0
    assert stats()["misses"] == 0

