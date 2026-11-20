import pytest
from langchain_core.documents import Document
from src.retrieval import fuse_results

def test_fuse_results_ranking_order():
    doc_a = Document(page_content="Document A", metadata={"chunk_id": "chunk_A"})
    doc_b = Document(page_content="Document B", metadata={"chunk_id": "chunk_B"})
    doc_c = Document(page_content="Document C", metadata={"chunk_id": "chunk_C"})
    
    # List 1: A ranked #1, B ranked #2
    list_1 = [doc_a, doc_b]
    # List 2: A ranked #1, C ranked #2
    list_2 = [doc_a, doc_c]
    
    # Doc A appears at rank 0 in both lists -> should have highest RRF score
    fused = fuse_results([list_1, list_2], top_k=3)
    
    assert len(fused) == 3
    assert fused[0].metadata["chunk_id"] == "chunk_A"
    
def test_fuse_results_top_k_truncation():
    docs = [
        Document(page_content=f"Doc {i}", metadata={"chunk_id": f"id_{i}"})
        for i in range(10)
    ]
    fused = fuse_results([docs], top_k=4)
    assert len(fused) == 4

