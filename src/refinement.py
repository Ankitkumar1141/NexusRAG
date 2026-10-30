import logging
from typing import List, Tuple
from langchain_core.documents import Document
import config

logger = logging.getLogger(__name__)


def describe_source(document: Document) -> str:
    filename = document.metadata.get("filename", "unknown file")

    page_number = document.metadata.get("page_number")
    if page_number:
        return f"{filename} - Page {page_number}"

    section = document.metadata.get("section")
    if section:
        return f"{filename} - Section: {section}"

    return filename


def drop_irrelevant(documents: List[Document]) -> List[Document]:
    relevant = [document for document in documents
        if document.metadata.get("rerank_score", 0.0) >= config.MIN_RERANK_SCORE]

    if not relevant and documents:
        logger.info("All chunks scored below the threshold; keeping the best one.")
        return documents[:1]

    return relevant


def apply_context_budget(documents: List[Document]) -> List[Document]:
    selected: List[Document] = []
    used_characters = 0

    for document in documents:
        length = len(document.page_content)
        if used_characters + length > config.MAX_CONTEXT_CHARS:
            continue
        selected.append(document)
        used_characters += length
    if not selected and documents:
        trimmed = Document(
            page_content=documents[0].page_content[: config.MAX_CONTEXT_CHARS],
            metadata=documents[0].metadata,
        )
        selected.append(trimmed)

    return selected


def format_context(documents: List[Document]) -> str:
    blocks = []
    for number, document in enumerate(documents, start=1):
        blocks.append(
            f"[{number}] Source: {describe_source(document)}\n{document.page_content}"
        )
    return "\n\n".join(blocks)


def refine_context(documents: List[Document]) -> Tuple[List[Document], str]:
    relevant = drop_irrelevant(documents)
    selected = apply_context_budget(relevant)
    return selected, format_context(selected)