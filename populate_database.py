import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from unstructured.documents.elements import Element

from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from unstructured.partition.pdf import partition_pdf


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    for root, dirs, files in os.walk(DATA_PATH):
        for filename in files:
            if filename.endswith(".pdf"):
                file_path = os.path.join(root, filename)
                print(f"Processing {file_path}")
                elements = partition_pdf(file_path)
                add_to_chroma(elements)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Element]):
    # Load the existing database.
    db = Chroma(
        collection_name='iollama', persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(),
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    texts = []
    metadata = []
    ids = []
    for chunk in chunks_with_ids:
        if chunk["id"] not in existing_ids:
            texts.append(chunk['text'])
            metadata.append(chunk['metadata'])
            ids.append(chunk['id'])

    if len(texts):
        print(f"👉 Adding new documents: {len(texts)}")
        db.add_texts(texts, metadata, ids)
        db.persist()
        print("✅ Added all new documents")
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0
    new_chunks = []
    for chunk in chunks:

        source = chunk.metadata.filename
        page = chunk.metadata.page_number
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        # Add it to the page meta-data.
        new_chunks.append({'id': chunk_id, 'text': f'<{chunk.category}>{chunk.text}</{chunk.category}>',
                           'metadata': {'source': source, 'page': page, 'id': chunk_id}})

    return new_chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
