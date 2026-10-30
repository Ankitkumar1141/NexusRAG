
import logging
from typing import Dict, List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_client import get_chat_model
from src.chunking import build_parent_lookup
from src.indexing import load_doc_store, load_vector_store
from src.memory import format_history, get_recent_messages
from src.refinement import describe_source, refine_context
from src.reranking import rerank_documents
from src.retrieval import (build_keyword_retriever, generate_query_variants, hybrid_retrieve, recover_parents,)

logger = logging.getLogger(__name__)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions about internal "
            "company documents.\n\n"
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


def answer_question(question: str,pipeline: Dict[str, object],memory: InMemoryChatMessageHistory,) -> Dict[str, object]:
    if not question or not question.strip():
        raise ValueError("Please type a question first.")

    
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

    return {
        "answer": answer,
        "sources": sources,
        "debug": {
            "original_query": question,
            "generated_queries": queries,
            "retrieved": summarise_documents(candidate_children),
            "reranked": summarise_documents(reranked),
            "final_context": summarise_documents(selected),
        },
    }


def generate_answer(question: str,context_text: str,memory: InMemoryChatMessageHistory,) -> str:
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
        logger.error("Answer generation failed: %s", error)
        return (
            "Sorry, the OpenAI request failed. Check your API key, your "
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