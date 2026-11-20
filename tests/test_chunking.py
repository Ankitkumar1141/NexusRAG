import pytest
from langchain_core.documents import Document
from src.chunking import make_splitter, split_documents, build_parent_lookup

def test_make_splitter():
    splitter = make_splitter(chunk_size=100, chunk_overlap=20)
    text = (
        "Paragraph one is descriptive and contains several introductory sentences about the platform.\n\n"
        "Paragraph two continues with technical specifications regarding maximum memory limits and timeouts.\n\n"
        "Paragraph three summarizes the deployment workflow for enterprise cloud customers."
    )
    chunks = splitter.split_text(text)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 120

def test_split_documents_hierarchical():
    long_content = (
        "1. Working Hours Policy\n\n"
        "TechNova operates on flexible hours. Employees are expected to be available "
        "during the core hours of 10:00 to 16:00 in their local time zone, and to work "
        "an average of 40 hours per week.\n\n"
        "Teams hold a daily stand-up inside core hours. Meetings outside core hours "
        "should be recorded so colleagues in other regions can catch up easily."
    )
    doc = Document(page_content=long_content, metadata={"filename": "handbook.pdf", "doc_type": "pdf"})
    
    parents, children = split_documents([doc])
    
    assert len(parents) >= 1
    assert len(children) >= 1
    
    # Check parent metadata
    parent = parents[0]
    assert "parent_id" in parent.metadata
    assert parent.metadata["filename"] == "handbook.pdf"
    assert "section" in parent.metadata
    
    # Check child metadata lineage
    child = children[0]
    assert "parent_id" in child.metadata
    assert "chunk_id" in child.metadata
    assert child.metadata["parent_id"] == parent.metadata["parent_id"]

def test_build_parent_lookup():
    p1 = Document(page_content="Parent 1", metadata={"parent_id": "doc1 :: 0"})
    p2 = Document(page_content="Parent 2", metadata={"parent_id": "doc1 :: 1"})
    
    lookup = build_parent_lookup([p1, p2])
    assert len(lookup) == 2
    assert lookup["doc1 :: 0"].page_content == "Parent 1"
    assert lookup["doc1 :: 1"].page_content == "Parent 2"

