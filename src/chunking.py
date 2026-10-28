import config
from typing import List, Dict, Tuple
from langchain_core.documents import Document

## import RecursiveCharacterTextSplitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except OSError as e:
    raise ImportError(f"Error while import in chunking module: {e}")

## import guess_section method from src/ingestion.py
from src.ingestion import guess_section


## make_splitter
def make_splitter(chunk_size: int, chunk_overlap: int) ->RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        length_function = len,)

## split_documents
def split_documents(documents: List[Document]):
    parent_splitter = make_splitter(config.PARENT_CHUNK_SIZE, config.PARENT_CHUNK_OVERLAP)  ## Parent splitter
    child_splitter  = make_splitter(config.CHILD_CHUNK_SIZE, config.CHILD_CHUNK_OVERLAP) ## Child splitter
    parents = []  ## list of parent chunks
    children = []  ##list of child chunks
    parents_per_file = {}

    for doc in documents:
        filename = doc.metadata.get("filename","unknown")

        for parent_text in parent_splitter.split_text(doc.page_content):
            parent_number = parents_per_file.get(filename,0)
            parents_per_file[filename] = parent_number + 1
            parent_id = f"{filename} :: {parent_number}"

            parent_metadata = dict(doc.metadata)
            parent_metadata["parent_id"] = parent_id
            parent_metadata["section"] = guess_section(parent_text)

            parents.append(Document(page_content = parent_text,metadata = parent_metadata))

            for child_index,child_text in enumerate(child_splitter.split_text(parent_text)):
                child_metadata = dict(parent_metadata)
                child_metadata["chunk_id"] = f"{parent_id} :: {child_index}"
                children.append(Document(page_content=child_text,metadata = child_metadata))
    return parents, children

## mapping of parent_id to the relevant chunk
def build_parent_lookup(parents: List[Document]) -> Dict[str, Document]:
    return {parent.metadata["parent_id"]: parent for parent in parents}           