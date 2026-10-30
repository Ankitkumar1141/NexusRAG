import json
import logging
import shutil
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

import config
from llm_client import get_embedding_model
from src.chunking import split_documents

logger = logging.getLogger(__name__)

def _documents_to_dicts(documents: List[Document]) -> List[dict]:
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]


def _dicts_to_documents(items: List[dict]) -> List[Document]:

    documents = []

    for item in items:

        if not isinstance(item, dict):
            logger.warning("Skipping invalid document entry: %s", type(item))
            continue

        page_content = item.get("page_content", "")

        if not page_content:
            logger.warning("Skipping document with missing page_content")
            continue

        metadata = item.get("metadata", {})

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata
            )
        )

    return documents


def save_doc_store(parents: List[Document], children: List[Document]) -> None:
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "parents": _documents_to_dicts(parents),
        "children": _documents_to_dicts(children),
    }
    config.DOC_STORE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Saved %d parents and %d children to %s",
        len(parents),
        len(children),
        config.DOC_STORE_PATH,
    )


def load_doc_store() -> Tuple[List[Document], List[Document]]:
    if not config.DOC_STORE_PATH.exists():
        return [], []

    payload = json.loads(config.DOC_STORE_PATH.read_text(encoding="utf-8"))
    parents = _dicts_to_documents(payload.get("parents", []))
    children = _dicts_to_documents(payload.get("children", []))
    return parents, children

def create_vector_store(children: List[Document]) -> VectorStore:
    if not children:
        raise ValueError(
            "There are no chunks to index. Add documents to data/documents "
            "or upload a file before building the index."
        )

    embeddings = get_embedding_model()

    if config.VECTOR_DB == "faiss":
        # from_documents() calls the OpenAI embedding API for every chunk and
        # stores the resulting vectors in memory; save_local() writes them to
        # vectorstore/faiss/ so we do not pay to embed them again.
        store = _faiss_class().from_documents(children, embeddings)
        config.FAISS_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(config.FAISS_DIR))
        return store

    if config.VECTOR_DB == "pinecone":
        config.check_pinecone_config()
        PineconeVectorStore = _pinecone_class()
        return PineconeVectorStore.from_documents(
            children,
            embeddings,
            index_name=config.PINECONE_INDEX_NAME,
            namespace=config.PINECONE_NAMESPACE,
        )

    raise ValueError(
        f"Unknown VECTOR_DB '{config.VECTOR_DB}'. Use 'faiss' or 'pinecone'."
    )


def load_vector_store() -> VectorStore:
    embeddings = get_embedding_model()

    if config.VECTOR_DB == "faiss":
        if not (config.FAISS_DIR / "index.faiss").exists():
            raise FileNotFoundError(
                "No FAISS index found. Click 'Build Vector DB' in the sidebar "
                "or run: python scripts/build_index.py"
            )
        # allow_dangerous_deserialization is required because FAISS indexes are
        # loaded with pickle. It is safe here: we are loading a file this same
        # app created on this same machine.
        return _faiss_class().load_local(
            str(config.FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if config.VECTOR_DB == "pinecone":
        config.check_pinecone_config()
        PineconeVectorStore = _pinecone_class()
        return PineconeVectorStore.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=config.PINECONE_NAMESPACE,
        )

    raise ValueError(
        f"Unknown VECTOR_DB '{config.VECTOR_DB}'. Use 'faiss' or 'pinecone'."
    )


def _faiss_class():
    from langchain_community.vectorstores import FAISS

    return FAISS


def _pinecone_class():
    try:
        from langchain_pinecone import PineconeVectorStore
    except ImportError as error:
        raise ImportError(
            "VECTOR_DB=pinecone needs the Pinecone package.\n"
            "Install it with: pip install langchain-pinecone"
        ) from error

    return PineconeVectorStore



def build_index(documents: List[Document]) -> Dict[str, int]:
    if not documents:
        raise ValueError("No documents to index. Add files to data/documents first.")

    parents, children = split_documents(documents)

    # Wipe the old index first so a rebuild cannot leave stale vectors behind.
    clear_index()

    create_vector_store(children)
    save_doc_store(parents, children)

    return {
        "documents": len(documents),
        "parents": len(parents),
        "children": len(children),
    }


def update_index(documents: List[Document]) -> Dict[str, int]:
    existing_parents, existing_children = load_doc_store()
    if not existing_children:
        # Nothing indexed yet, so "update" is really just "build".
        return build_index(documents)

    new_parents, new_children = split_documents(documents)

    # Chunk ids are built from the filename, so re-uploading the same file
    # produces the same ids. Skipping them prevents duplicate search results.
    # Side effect worth knowing: if you EDIT a file and keep its name, the
    # update will skip it. Use a full rebuild after editing a document.
    known_chunk_ids = {doc.metadata["chunk_id"] for doc in existing_children}
    known_parent_ids = {doc.metadata["parent_id"] for doc in existing_parents}

    children_to_add = [
        doc for doc in new_children if doc.metadata["chunk_id"] not in known_chunk_ids
    ]
    parents_to_add = [
        doc for doc in new_parents if doc.metadata["parent_id"] not in known_parent_ids
    ]

    if not children_to_add:
        return {"added_children": 0, "skipped_children": len(new_children)}

    store = load_vector_store()
    if config.VECTOR_DB == "pinecone":
        store.add_documents(children_to_add, namespace=config.PINECONE_NAMESPACE)
    else:
        store.add_documents(children_to_add)
        store.save_local(str(config.FAISS_DIR))

    save_doc_store(
        existing_parents + parents_to_add,
        existing_children + children_to_add,
    )

    return {
        "added_children": len(children_to_add),
        "skipped_children": len(new_children) - len(children_to_add),
    }


def clear_index() -> None:
    if config.FAISS_DIR.exists():
        shutil.rmtree(config.FAISS_DIR)

    if config.DOC_STORE_PATH.exists():
        config.DOC_STORE_PATH.unlink()

    if config.VECTOR_DB == "pinecone":
        _clear_pinecone_namespace()

    logger.info("Index cleared.")


def _clear_pinecone_namespace() -> None:
    try:
        from pinecone import Pinecone

        pinecone = Pinecone(api_key=config.PINECONE_API_KEY)
        index = pinecone.Index(config.PINECONE_INDEX_NAME)
        index.delete(delete_all=True, namespace=config.PINECONE_NAMESPACE)
    except Exception as error:
        # An empty or missing namespace raises an error we can safely ignore.
        logger.warning("Could not clear Pinecone namespace: %s", error)


def index_exists() -> bool:
    if not config.DOC_STORE_PATH.exists():
        return False

    if config.VECTOR_DB == "faiss":
        return (config.FAISS_DIR / "index.faiss").exists()

    return True


def index_summary() -> Dict[str, object]:
    parents, children = load_doc_store()
    filenames = sorted({doc.metadata.get("filename", "?") for doc in parents})
    return {
        "parents": len(parents),
        "children": len(children),
        "files": filenames,
    }