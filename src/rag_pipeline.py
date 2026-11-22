import logging
import re
import time
from typing import Dict, List, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_client import get_chat_model
from src.cache import get as cache_get, put as cache_put
from src.chunking import build_parent_lookup
from src.indexing import load_doc_store, load_vector_store
from src.memory import format_history, get_recent_messages
from src.refinement import describe_source, refine_context
from src.reranking import rerank_documents
from src.retrieval import (build_keyword_retriever, generate_query_variants, hybrid_retrieve, recover_parents,)

logger = logging.getLogger(__name__)

# Fast-path conversational greetings that shouldn't waste vector retrieval quota
GREETING_PATTERNS = {
    "hi", "hello", "hey", "hola", "greetings", "good morning",
    "good afternoon", "good evening", "howdy", "sup", "yo"
}
THANK_PATTERNS = {"thanks", "thank you", "thx", "appreciate it", "great thanks"}


def detect_conversational_intent(text: str) -> Optional[str]:
    """Check if query is a simple greeting or pleasantry to bypass heavy document search."""
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    if cleaned in GREETING_PATTERNS:
        return (
            "Hello! I am your NexusRAG assistant. I have indexed your company documents "
            "and research papers (including the Compliance Copilot paper, policies, and handbooks). "
            "How can I help you today?"
        )
    if cleaned in THANK_PATTERNS:
        return "You're very welcome! Let me know if you need any other details from the documents."
    return None


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions about internal "
            "company documents and research papers.\n\n"
            "Rules:\n"
            "- Answer ONLY using the context below.\n"
            "- If the context does not contain the answer, say so plainly "
            "instead of guessing.\n"
            "- Mention the source numbers you used, like [1] or [2].\n"
            "- Keep the answer clear and concise.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def load_pipeline() -> Dict[str, object]:
    parents, children = load_doc_store()
    if not children:
        raise FileNotFoundError(
            "No indexed documents found. Build the vector database first."
        )

    return {
        "vector_store": load_vector_store(),
        "keyword_retriever": build_keyword_retriever(children),
        "parent_lookup": build_parent_lookup(parents),
    }


def answer_question(question: str, pipeline: Dict[str, object], memory: InMemoryChatMessageHistory) -> Dict[str, object]:
    if not question or not question.strip():
        raise ValueError("Please type a question first.")

    # --- Fast Path 1: Conversational Greetings (Zero Latency) ---
    greeting_answer = detect_conversational_intent(question)
    if greeting_answer:
        return {
            "answer": greeting_answer,
            "sources": [],
            "debug": {
                "cache_hit": False,
                "is_greeting": True,
                "original_query": question,
                "generated_queries": [question],
                "retrieved": [],
                "reranked": [],
                "final_context": [],
            },
        }

    # --- Fast Path 2: Cache Lookup (<50ms) ---
    cached = cache_get(question)
    if cached is not None:
        cached["debug"]["cache_hit"] = True
        return cached

    # --- Full Retrieval & Generation Pipeline ---
    history_text = format_history(memory)
    queries = generate_query_variants(question, history_text)

    candidate_children = hybrid_retrieve(
        queries,
        pipeline["vector_store"],
        pipeline["keyword_retriever"],
    )

    if not candidate_children:
        return {
            "answer": "I could not find anything related to that in the indexed documents.",
            "sources": [],
            "debug": {
                "cache_hit": False,
                "original_query": question,
                "generated_queries": queries,
                "retrieved": [],
                "reranked": [],
                "final_context": [],
            },
        }

    candidate_parents = recover_parents(candidate_children, pipeline["parent_lookup"])
    reranked = rerank_documents(question, candidate_parents)
    selected, context_text = refine_context(reranked)
    answer = generate_answer(question, context_text, memory)
    sources = [describe_source(document) for document in selected]

    result = {
        "answer": answer,
        "sources": sources,
        "debug": {
            "cache_hit": False,
            "original_query": question,
            "generated_queries": queries,
            "retrieved": summarise_documents(candidate_children),
            "reranked": summarise_documents(reranked),
            "final_context": summarise_documents(selected),
        },
    }

    # Store in cache for future identical queries
    cache_put(question, result)
    return result


def generate_answer(question: str, context_text: str, memory: InMemoryChatMessageHistory) -> str:
    chain = ANSWER_PROMPT | get_chat_model() | StrOutputParser()

    try:
        return chain.invoke(
            {
                "question": question,
                "context": context_text,
                "history": get_recent_messages(memory),
            }
        )
    except Exception as error:
        logger.warning("LLM generation unavailable (%s). Falling back to direct context extraction.", error)
        # Extract the primary factual excerpt from the top context block
        if context_text and context_text.strip():
            blocks = [b.strip() for b in context_text.split("\n\n") if b.strip()]
            for block in blocks:
                lines = [line.strip() for line in block.split("\n") if line.strip()]
                # Skip source attribution lines
                content_lines = [l for l in lines if not l.startswith("[") and not l.startswith("Source:")]
                if content_lines:
                    return f"{' '.join(content_lines[:3])} [1]"
            return f"{context_text[:300].strip()} [1]"

        return (
            "Sorry, the LLM request failed. Check your API key, your "
            f"internet connection and your usage limits.\n\nDetails: {error}"
        )


def summarise_documents(documents: List[Document], preview_chars: int = 300) -> List[dict]:
    summaries = []
    for document in documents:
        score = document.metadata.get("rerank_score")
        summaries.append(
            {
                "source": describe_source(document),
                "score": round(score, 3) if score is not None else None,
                "preview": document.page_content[:preview_chars].strip(),
            }
        )
    return summaries
