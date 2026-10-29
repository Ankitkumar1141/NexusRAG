import json
import logging
import shutil
import config
from typing import List,Dict, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

## import from src folder
from llm_client import get_embedding_model
from src.chunking import split_documents

logger = logging.getLogger(__name__)

def _documents_to_dict(doc: List[Document]) -> List[Dict]:
    return [
        {
            "page_content": docs.page_content,
            "meta_data": docs.metadata
        }
        for docs in doc
    ]

def _dict_to_documents(dict: List[Dict]) ->List[Document]:
    return [
        Document(page_content = item["page_content"], metadata = item["metadata"])
        for item in dict
    ]

def save_doc_store(parents: List[Document], child: List[Document]):
    config.VECTORSTORE_DIR.mkdir(parents=True,exist_ok=True)
    payload = {
        "parents": _documents_to_dict(parents),
        "child": _documents_to_dict(child)
    }
    config.DOC_STORE_PATH.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    logger.info("Saved %d parents and %d chilren to %s",
                len(parents),
                len(child),
                config.DOC_STORE_PATH)


def load_doc_store():
    if not config.DOC_STORE_PATH.exists():
        return [],[]
    payload = json.loads(config.DOC_STORE_PATH.read_text(encoding="utf-8"))
    parents = _dict_to_documents(payload.get("parents",[]))
    children = _dict_to_documents(payload.get("children",[]))
    return parents, children

def create_vector_store(child: List[Document])-> VectorStore:
    if not child:
        raise ValueError(
            "There are no chunks to index. Add documents to data/documents or upload a file before building the index"
        )
    embedding = get_embedding_model()
    if config.VECTOR_DB == "faiss":
        store = _faiss_class().from_documents(child, embedding)
        config.FAISS_DIR.mkdir(parents=True,exist_ok=True)
        store.save_local(str(config.FAISS_DIR))
        return store
    if config.VECTOR_DB == "pinecone":
        config.check_pinecone_config()
        PineconeVectorStore = _pinecone_class()
        return PineconeVectorStore.from_documents(
            child,
            embedding,
            index_name = config.PINECONE_INDEX_NAME,
            namespace = config.PINECONE_NAMESPACE
        )
    raise ValueError(
        f"Unknown VECTOR_DB '{config.VECTOR_DB}'. Please use either faiss or pinecone."
    )

def load_vector_store():
    embeddings = get_embedding_model()    
    if config.VECTOR_DB == "faiss":
        if not (config.FAISS_DIR / "index.faiss").exists():
            raise FileNotFoundError(
                "No FAISS index found. Click 'Build VectorDB' in the sidebar or run: python scripts/build_index.py."
            )
        return _faiss_class().load_local(
            str(config.FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization = True,
        )

    if config.VECTOR_DB == "pinecone":
        config.check_pinecone_config()
        PineconeVectorStore = _pinecone_class()
        return PineconeVectorStore.from_existing_index(
            index_name = config.PINECONE_INDEX_NAME,
            embedding = embeddings,
            namespace = config.PINECONE_NAMESPACE,
        )

    raise ValueError(f"Unknown VECTOR_DB '{config.VECTOR_DB}'. Use faiss or piencone please.")

def _faiss_class():
    from langchain_community.vectorstores import FAISS
    return FAISS

def _pinecone_class():
    try:
        from langchain_pinecone import PineconeVectorStore
    except ImportError as e:
        raise ImportError("VECTOR_DB=pinecone needs the Pinecone packed, do pip install it.") 


def build_index(documents: List[Document])-> Dict[str,int]:
    if not documents:
        raise ValueError("No documents to index available. Add files to data/documents first.")
    parents, children = split_documents(documents)
    clear_index()

    create_vector_store(children)
    save_doc_store(parents, children)

    return {
        "documents": len(documents),
        "parents": len(parents),
        "children": len(children),
    }

def update_index(documents: List[Document]):
    existing_parents, existing_children = load_vector_store()    
    if not existing_children:
        return build_index(documents)

    new_parents, new_children = split_documents(documents)

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
    """
    Delete everything we have indexed so the next build starts clean.

    Useful when you change the embedding model, change the chunk sizes, or
    simply want to start over with a different set of documents.
    """    
    if config.FAISS_DIR.exists():
        shutil.rmtree(config.FAISS_DIR)

    if config.DOC_STORE_PATH.exists():
        config.DOC_STORE_PATH.unlink()

    if config.VECTOR_DB == "pinecone":
        _clear_pinecone_namespace()

    logger.info("index cleared.")

def _clear_pinecone_namespace():
    try:
        from pinecone import Pinecone

        pinecone = Pinecone(api_key = config.PINECONE_API_KEY)    
        index = pinecone.Index(config.PINECONE_INDEX_NAME)
        index.delete(delete_all = True, namespace = config.PINECONE_NAMESPACE)

    except Exception as e:
        logger.warning("Could not clear Pinecone namespace: %s", e)

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