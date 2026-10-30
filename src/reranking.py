import logging
import re
from collections import Counter
from functools import lru_cache
from typing import List

from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)

_cross_encoder_works = True
_fallback_reason = ""
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
}


@lru_cache(maxsize=1)
def get_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is not installed "
            "(pip install sentence-transformers)"
        ) from error

    logger.info("Loading cross-encoder: %s", config.CROSS_ENCODER_MODEL)
    return CrossEncoder(config.CROSS_ENCODER_MODEL)


def rerank_documents(query: str,documents: List[Document],top_k: int | None = None,) -> List[Document]:
    global _cross_encoder_works, _fallback_reason

    if not documents:
        return []

    keep = top_k or config.RERANK_TOP_K
    if config.RERANKER == "simple" or not _cross_encoder_works:
        return keyword_rerank(query, documents, keep)

    try:
        model = get_cross_encoder()
        pairs = [(query, document.page_content) for document in documents]
        scores = model.predict(pairs)
    except Exception as error:
        _cross_encoder_works = False
        _fallback_reason = str(error).split("\n")[0]
        logger.warning(
            "Cross-encoder unavailable (%s). Falling back to the simple "
            "keyword reranker. Set RERANKER=simple in .env to silence this.",
            _fallback_reason,
        )
        return keyword_rerank(query, documents, keep)

    for document, score in zip(documents, scores):
        document.metadata["rerank_score"] = float(score)

    return sort_by_score(documents, keep)


def keyword_rerank(query: str, documents: List[Document], keep: int) -> List[Document]:
    for document in documents:
        document.metadata["rerank_score"] = keyword_score(query, document.page_content)

    return sort_by_score(documents, keep)


def keyword_score(query: str, text: str) -> float:
    query_words = tokenize(query)
    if not query_words:
        return 0.0

    found = Counter(
        word for word in re.findall(r"[a-z0-9]+", text.lower()) if word in query_words
    )
    coverage = len(found) / len(query_words)
    repeats = min(sum(found.values()) - len(found), 10) * 0.08

    return 10.0 * coverage - 5.0 + repeats


def tokenize(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if len(word) > 1 and word not in STOP_WORDS}


def sort_by_score(documents: List[Document], keep: int) -> List[Document]:
    ranked = sorted(documents,key=lambda document: document.metadata.get("rerank_score", 0.0),reverse=True,)
    return ranked[:keep]


def reranker_status() -> str:
    if config.RERANKER == "simple":
        return "simple keyword (set by RERANKER=simple)"
    if not _cross_encoder_works:
        return f"simple keyword (cross-encoder failed: {_fallback_reason[:60]})"
    return f"cross-encoder ({config.CROSS_ENCODER_MODEL})"
