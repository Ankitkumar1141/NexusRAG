import os
import re
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf",".docx",".txt"}


## method to clean text
def clean_text(text: str) -> str:
    ## check for empty text
    if not text:
        return ""

    ## text cleaning
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

## method to add meta data
def add_meta_data(documents:List[Document], path: Path) -> List[Document]:
    
    ## adding metadata to the document where metadata includes - source, filename, doc_type
    for document in documents:
        document.metadata["source"] = str(path)
        document.metadata["filename"] = path.name
        document.metadata["doc_type"] = path.suffix.lower().lstrip(".")

        ## adding page number
        if "page" in document.metadata:
            document.metadata["page_number"] = int(document.metadata["page"])+1

    return documents

## document loader using file path
def load_document(file_path: str|Path) ->List[Document]:
    path = Path(file_path)
    extension = path.suffix.lower()

    ## check for unsupported extensions
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}'."
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    ## check for path
    if not path.exists():
        raise FileNotFoundError(f"File not present at {path}")

    ## pdf_loader --> pdf
    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
    ## text_loader --> .txt
    elif extension == ".txt":
        loader = TextLoader(file_path=str(path), encoding="utf-8")
    ## docx_loader --> .docx
    else:
        loader = Docx2txtLoader(str(path))

    documents = loader.load()

    cleaned_documents = []

    ## iterate throughout the list of documents to clean the text
    for doc in documents:

        ## apply clean_text method to every doc page_content
        doc.page_content = clean_text(doc.page_content)

        if doc.page_content.strip():
            cleaned_documents.append(doc)

    if not cleaned_documents:
        logger.warning("No readable text found in %s. Skipping it",path.name)

    return add_meta_data(cleaned_documents,path)



## load the document from folder
def load_document_from_folder(folder_path: Path):
    path = Path(folder_path)

    ## check for folder path
    if not path.exists():
        logger.warning("Document fodler does not exist: %s", path)
        return []

    all_documents: List[Document] = []
    ## iterate throughout the folder to check file extensions
    for file_path in sorted (path.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            all_documents.extend(load_document(file_path))
        except Exception as e:
            logger.warning("Could not load %s: %s", file_path.name, e)

    logger.info("Loaded %d document pages from %s", len(all_documents), path)
    return all_documents 


## method to add section
def guess_section(text: str) -> str:
    for raw_line in text.split("\n")[:15]:
        line = raw_line.strip()
        if not (3 < len(line) <= 70) or line.endswith((".", ",", ";", ":")):
            continue
        if len(line.split()) > 8:
            continue

        looks_numbered = re.match(r"^\d+(\.\d+)*[.)]?\s+\S", line) is not None
        if line.isupper() or line.istitle() or looks_numbered:
            return re.sub(r"^\d+(\.\d+)*[.)]?\s+", "", line).strip()
        
    return ""

