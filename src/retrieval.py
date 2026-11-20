"""
Advanced techniques implemented are:

1. Multi-Query Retrieval - the LLM rewrites the question a few different ways.
2. Hybrid Retrieval      - concurrent meaning-based (vector) and keyword-based (BM25) search.
3. Parent-Child Recovery - we swap each matched small chunk for its big parent.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

import config
from llm_client import get_chat_model

logger = logging.getLogger(__name__)

# In-memory query-level vector search cache to prevent duplicate Gemini embedding calls
_vector_cache: Dict[Tuple[str, int], List[Document]] = {}


# ---------------------------------------------------------------------------
# 1. Multi-Query Retrieval
# ---------------------------------------------------------------------------

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite search queries for a document search engine.\n"
            "Given a user question, write {num_variants} alternative versions "
            "of it that use different wording or synonyms but keep the same "
            "meaning.\n"
            "If conversation history is given, make each version standalone by "
            "replacing words like 'it' or 'that policy' with the real subject.\n"
            "Return one query per line. No numbering, no extra text.",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\nUser question: {question}",
        ),
    ]
)


def generate_query_variants(question: str, history: str = "") -> List[str]:
    # For very short queries, multi-query is unnecessary and consumes quota
    if len(question.strip().split()) <= 2:
        return [question]

    queries = [question]

    try:
        chain = MULTI_QUERY_PROMPT | get_chat_model() | StrOutputParser()
        response = chain.invoke(
            {
                "question": question,
                "history": history or "(none)",
                "num_variants": config.NUM_QUERY_VARIANTS,
            }
        )
        for line in response.split("\n"):
            cleaned = line.strip(" -*0123456789.").strip()
            if cleaned and cleaned.lower() not in [q.lower() for q in queries]:
                queries.append(cleaned)
    except Exception as error:
        # If the rewrite call fails we still have the original question, so
        # the user gets an answer instead of an error screen.
        logger.warning("Multi-query generation failed (%s). Using original query.", error)

    # Cap to avoid rate limit spikes
    return queries[: config.NUM_QUERY_VARIANTS + 1]


# ---------------------------------------------------------------------------
# 2. Hybrid Retrieval (vector search + BM25 keyword search)
# ---------------------------------------------------------------------------

def build_keyword_retriever(children: List[Document]) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(children)
    retriever.k = config.BM25_TOP_K
    return retriever


def search_vector_store(store: VectorStore, query: str, k: int) -> List[Document]:
    cache_key = (query.strip().lower(), k)
    if cache_key in _vector_cache:
        return _vector_cache[cache_key]

    try:
        results = store.similarity_search(query, k=k)
        if len(_vector_cache) < 256:
            _vector_cache[cache_key] = results
        return results
    except Exception as error:
        logger.warning("Vector search failed for '%s': %s", query, error)
        return []


def search_keywords(retriever: BM25Retriever, query: str) -> List[Document]:
    try:
        return retriever.invoke(query)
    except Exception as error:
        logger.warning("Keyword search failed for '%s': %s", query, error)
        return []


def fuse_results(ranked_lists: List[List[Document]], top_k: int) -> List[Document]:
    # The constant 60 comes from the original RRF paper. It stops the very
    # first result from completely dominating everything below it.
    rrf_constant = 60

    scores: Dict[str, float] = {}
    documents: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for position, document in enumerate(ranked_list):
            chunk_id = document.metadata.get("chunk_id", document.page_content[:80])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_constant + position)
            documents[chunk_id] = document

    best_ids = sorted(scores, key=lambda key: scores[key], reverse=True)[:top_k]
    return [documents[chunk_id] for chunk_id in best_ids]


def hybrid_retrieve(queries: List[str], store: VectorStore, keyword_retriever: BM25Retriever) -> List[Document]:
    """Execute vector and BM25 searches across all query variants concurrently in parallel."""
    ranked_lists: List[List[Document]] = []

    if not queries:
        return []

    # Deduplicate queries to avoid redundant embedding calls
    seen = set()
    unique_queries = []
    for q in queries:
        normalized = q.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_queries.append(q)

    # Parallelize I/O-bound vector embedding lookups and in-memory BM25 matching
    max_workers = min(len(unique_queries) * 2, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        vector_futures = [
            executor.submit(search_vector_store, store, q, config.VECTOR_TOP_K)
            for q in unique_queries
        ]
        keyword_futures = [
            executor.submit(search_keywords, keyword_retriever, q)
            for q in unique_queries
        ]

        for future in vector_futures:
            try:
                ranked_lists.append(future.result())
            except Exception as e:
                logger.warning("Error collecting parallel vector search: %s", e)

        for future in keyword_futures:
            try:
                ranked_lists.append(future.result())
            except Exception as e:
                logger.warning("Error collecting parallel keyword search: %s", e)

    return fuse_results(ranked_lists, config.HYBRID_TOP_K)


# ---------------------------------------------------------------------------
# 3. Parent-Child Recovery
# ---------------------------------------------------------------------------

def recover_parents(children: List[Document], parent_lookup: Dict[str, Document]) -> List[Document]:
    parents: List[Document] = []
    seen_parent_ids = set()

    for child in children:
        parent_id = child.metadata.get("parent_id")
        parent = parent_lookup.get(parent_id)

        if parent is None:
            # No parent found (for example an index built by an older
            # version), so we keep the child rather than losing the result.
            parents.append(child)
            continue

        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            parents.append(parent)

    return parents
