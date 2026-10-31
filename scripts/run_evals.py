import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.memory import create_memory
from src.rag_pipeline import answer_question, load_pipeline
from src.refinement import describe_source, refine_context
from src.reranking import rerank_documents
from src.retrieval import hybrid_retrieve, recover_parents

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("evals")
logger.setLevel(logging.INFO)

GROUND_TRUTH_PATH = config.PROJECT_ROOT / "data" / "evaluation" / "ground_truth.json"
RESULTS_PATH = config.PROJECT_ROOT / "data" / "evaluation" / "eval_results.json"


def calculate_mrr(candidates: List[dict], expected_doc: str) -> float:
    """Calculate Reciprocal Rank (1/rank) for the first occurrence of expected_doc."""
    if not expected_doc:
        return 1.0
    for rank, cand in enumerate(candidates, start=1):
        source = cand.get("source", "")
        if expected_doc.lower() in source.lower():
            return 1.0 / rank
    return 0.0


def check_keywords(text: str, keywords: List[str]) -> float:
    """Calculate the fraction of expected keywords present in the text."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords)


def evaluate_retrieval_offline(question: str, pipeline: Dict[str, object]) -> Dict[str, Any]:
    """Offline retrieval evaluation using BM25 and vector store without external LLM rewrite."""
    candidate_children = hybrid_retrieve(
        [question],
        pipeline["vector_store"],
        pipeline["keyword_retriever"],
    )
    if not candidate_children:
        return {"retrieved": [], "final_context": [], "sources": [], "answer": "No documents found."}

    candidate_parents = recover_parents(candidate_children, pipeline["parent_lookup"])
    reranked = rerank_documents(question, candidate_parents)
    selected, context_text = refine_context(reranked)

    retrieved_summaries = [
        {"source": describe_source(doc), "preview": doc.page_content[:200]}
        for doc in candidate_children
    ]
    final_summaries = [
        {"source": describe_source(doc), "preview": doc.page_content[:200]}
        for doc in selected
    ]
    sources = [describe_source(doc) for doc in selected]

    return {
        "retrieved": retrieved_summaries,
        "final_context": final_summaries,
        "context_text": context_text,
        "sources": sources,
        "answer": context_text
    }


def run_evaluation(retrieval_only: bool = False) -> Dict[str, Any]:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth dataset not found at {GROUND_TRUTH_PATH}")

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    mode_label = "Retrieval & Ranking (Offline)" if retrieval_only else "End-to-End RAG (with LLM)"
    print(f"\n========================================================")
    print(f"  NexusRAG Quantitative Evaluation Benchmark")
    print(f"  Mode:    {mode_label}")
    print(f"  Dataset: {GROUND_TRUTH_PATH.name} ({len(test_cases)} test cases)")
    print(f"========================================================\n")

    pipeline = load_pipeline()

    eval_records = []
    total_retrieval_hits = 0
    total_context_hits = 0
    mrr_scores = []
    keyword_scores = []
    total_latency = 0.0

    for i, test in enumerate(test_cases, start=1):
        test_id = test.get("id", f"case_{i}")
        question = test["question"]
        expected_doc = test.get("expected_document")
        expected_section = test.get("expected_section")
        key_phrases = test.get("expected_key_phrases", [])
        is_negative_case = expected_doc is None

        start_time = time.perf_counter()
        if retrieval_only:
            result = evaluate_retrieval_offline(question, pipeline)
            answer = result["answer"]
            sources = result["sources"]
            retrieved = result["retrieved"]
            final_context = result["final_context"]
        else:
            memory = create_memory()
            res = answer_question(question, pipeline, memory)
            answer = res.get("answer", "")
            sources = res.get("sources", [])
            debug_info = res.get("debug", {})
            retrieved = debug_info.get("retrieved", [])
            final_context = debug_info.get("final_context", [])

        elapsed = time.perf_counter() - start_time
        total_latency += elapsed

        if is_negative_case:
            # For negative cases (out of scope), evaluate abstention
            abstain_keywords = ["not", "could not find", "unmentioned", "does not contain", "no information"]
            abstained = any(kw in answer.lower() for kw in abstain_keywords)
            retrieval_hit = True
            context_hit = True
            mrr = 1.0
            kw_score = 1.0 if abstained else 0.0
        else:
            # Check if expected document was retrieved
            retrieval_hit = any(expected_doc.lower() in cand.get("source", "").lower() for cand in retrieved)
            # Check if expected document survived reranking into final context
            context_hit = any(expected_doc.lower() in cand.get("source", "").lower() for cand in final_context)
            mrr = calculate_mrr(retrieved, expected_doc)
            # Check presence of key phrases in retrieved context or generated answer
            eval_target_text = answer if not retrieval_only else result.get("context_text", "")
            kw_score = check_keywords(eval_target_text, key_phrases)

        if retrieval_hit:
            total_retrieval_hits += 1
        if context_hit:
            total_context_hits += 1

        mrr_scores.append(mrr)
        keyword_scores.append(kw_score)

        status_mark = "PASS" if (context_hit and kw_score >= 0.5) else "FAIL"
        print(f"[{i:02d}/{len(test_cases):02d}] {status_mark} | ID: {test_id} | Latency: {elapsed:.2f}s | MRR: {mrr:.2f} | KeyCoverage: {kw_score * 100:.0f}%")
        print(f"      Q: {question[:75]}...")
        if not context_hit:
            print(f"      [!] Expected doc '{expected_doc}' missing in final context!")
        print()

        eval_records.append({
            "id": test_id,
            "question": question,
            "expected_document": expected_doc,
            "expected_section": expected_section,
            "latency_seconds": round(elapsed, 3),
            "retrieval_hit": retrieval_hit,
            "context_hit": context_hit,
            "reciprocal_rank": round(mrr, 3),
            "key_coverage": round(kw_score, 3),
            "sources_cited": sources
        })

    num_tests = len(test_cases)
    summary = {
        "mode": "retrieval_only" if retrieval_only else "end_to_end",
        "total_test_cases": num_tests,
        "retrieval_hit_rate": round(total_retrieval_hits / num_tests, 4),
        "context_precision_hit_rate": round(total_context_hits / num_tests, 4),
        "mean_reciprocal_rank_mrr": round(sum(mrr_scores) / num_tests, 4),
        "mean_key_phrase_coverage": round(sum(keyword_scores) / num_tests, 4),
        "avg_latency_seconds": round(total_latency / num_tests, 3)
    }

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "records": eval_records
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"========================================================")
    print(f"  EVALUATION SUMMARY REPORT")
    print(f"========================================================")
    print(f"  Total Test Cases           : {summary['total_test_cases']}")
    print(f"  Retrieval Hit Rate (Recall): {summary['retrieval_hit_rate'] * 100:.1f}%")
    print(f"  Context Hit Rate           : {summary['context_precision_hit_rate'] * 100:.1f}%")
    print(f"  Mean Reciprocal Rank (MRR) : {summary['mean_reciprocal_rank_mrr']:.3f}")
    print(f"  Key Coverage               : {summary['mean_key_phrase_coverage'] * 100:.1f}%")
    print(f"  Average Query Latency      : {summary['avg_latency_seconds']:.2f}s")
    print(f"========================================================")
    print(f"  Results saved to: {RESULTS_PATH}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NexusRAG quantitative evaluations")
    parser.add_argument("--retrieval-only", action="store_true", help="Run offline retrieval & reranking evals")
    args = parser.parse_args()

    run_evaluation(retrieval_only=args.retrieval_only)
