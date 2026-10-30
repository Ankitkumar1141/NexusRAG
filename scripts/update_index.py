import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  
from src.indexing import update_index  
from src.ingestion import load_documents_from_folder  

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    
    print(f"Reading documents from: {config.DOCUMENTS_DIR}")

    documents = load_documents_from_folder(config.DOCUMENTS_DIR)
    if not documents:
        print("No readable documents found. Add PDF, TXT or DOCX files and try again.")
        return

    try:
        stats = update_index(documents)
    except Exception as error:
        print(f"Update failed: {error}")
        return

    if "added_children" in stats:
        print(f"\nAdded {stats['added_children']} new child chunks.")
        print(f"Skipped {stats['skipped_children']} chunks that were already indexed.")
    else:
        # There was no index yet, so update_index() built one from scratch.
        print(f"\nNo index existed, so a new one was built with {stats['children']} chunks.")


if __name__ == "__main__":
    main()