import pytest
from langchain_core.documents import Document
from src.refinement import describe_source, drop_irrelevant, apply_context_budget, format_context
from src.reranking import keyword_score, tokenize, sort_by_score

def test_describe_source_with_page():
    doc = Document(page_content="Text", metadata={"filename": "policy.pdf", "page_number": 3})
    assert describe_source(doc) == "policy.pdf - Page 3"

def test_describe_source_with_section():
    doc = Document(page_content="Text", metadata={"filename": "guide.docx", "section": "Pricing"})
    assert describe_source(doc) == "guide.docx - Section: Pricing"

def test_drop_irrelevant_below_threshold():
    doc_good = Document(page_content="Good", metadata={"rerank_score": 1.5})
    doc_bad = Document(page_content="Bad", metadata={"rerank_score": -6.0})
    
    filtered = drop_irrelevant([doc_good, doc_bad])
    assert len(filtered) == 1
    assert filtered[0].page_content == "Good"

def test_drop_irrelevant_fallback_keeps_best():
    doc_bad1 = Document(page_content="Bad 1", metadata={"rerank_score": -5.0})
    doc_bad2 = Document(page_content="Bad 2", metadata={"rerank_score": -6.0})
    
    # If all score below threshold (-4.0), must keep the best one to avoid empty prompt
    filtered = drop_irrelevant([doc_bad1, doc_bad2])
    assert len(filtered) == 1
    assert filtered[0].page_content == "Bad 1"

def test_tokenize_filters_stop_words():
    words = tokenize("What is the annual leave policy for employees in 2026?")
    assert "what" not in words
    assert "the" not in words
    assert "annual" in words
    assert "leave" in words

def test_keyword_score():
    score_high = keyword_score("leave policy", "The annual leave policy gives 24 days.")
    score_low = keyword_score("leave policy", "This document covers database encryption keys.")
    assert score_high > score_low

