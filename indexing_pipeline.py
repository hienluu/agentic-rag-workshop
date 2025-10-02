# this file will be used to define the pipeline for indexing the markdown files, and store their chunks in the vector store
import argparse
import os
from vector_db import ChromaVectorDB
from markdown_splitter import split_markdown_file

def index_markdown_file(md_file: str, db: ChromaVectorDB):
    markdown_file = split_markdown_file(md_file)
    

def index_markdown_files(md_files: list[str], db: ChromaVectorDB):
    for md_file in md_files:
        index_markdown_file(md_file, db, embedding_model)

def main(collection_name: str, persist_directory: str, file: str, dir: str):
    db = ChromaVectorDB.create(collection_name, persist_directory)
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    embedding_model = EmbeddingModel.create(embedding_model_name)
    if file:
        index_markdown_file(file, db, embedding_model)
    elif dir:
        index_markdown_files(dir)
    else:
        print("No markdown files or directory to index.")
        raise ValueError("No markdown files or directory to index.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index the markdown files."
    )
    parser.add_argument("--collection_name", "-c", help="Vector database collection name to use", required=True)
    parser.add_argument("--persist_directory", "-p", help="Vector database persist directory to use", required=True)
    parser.add_argument("--file", "-f", help="Markdown file to index", required=False)
    parser.add_argument( "--dir", "-d", help="Directory containing the markdown files", required=False)
    args = parser.parse_args()
    print(args)
    main(args.file, args.dir)