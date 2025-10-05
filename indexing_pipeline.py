# this file will be used to define the pipeline for indexing the markdown files, and store their chunks in the vector store
import argparse
import os
from pathlib import Path
from vector_db import ChromaVectorDB
from embedding_model import EmbeddingModel
from markdown_splitter import split_markdown_file
# Load environment variables
from dotenv import load_dotenv

def index_markdown_file(md_file: str, vector_db: ChromaVectorDB, embedding_model: EmbeddingModel):
    print(f"========== index_markdown_file: {md_file} ==========")
    markdown_file_with_chunks = split_markdown_file(md_file)
    chunks = markdown_file_with_chunks.chunks
    
    collection_info = vector_db.get_collection_info()
    collection_name = collection_info['collection_name']
    document_count = collection_info['document_count']
 
    print(f"Start processing and adding {len(chunks)} chunks to collection {collection_name} with current document count {document_count}")
    # for each chunk, add the heading and the header text to the chunk before generating the embedding
    for i, chunk in enumerate(markdown_file_with_chunks.chunks): 
        #print(f"\n>>>> Processing chunk {i+1}:\n")
        page_content_with_heading = f"{chunk.metadata['heading']}\n\n{chunk.page_content}"

        embeddings = embedding_model.generate_embeddings(page_content_with_heading)
        metadata = {
            "url": markdown_file_with_chunks.page_info.page_url,
            "title": markdown_file_with_chunks.page_info.page_title,
            "chunk_index": i,
        }

        vector_db.add_documents([page_content_with_heading], 
                                [embeddings], 
                                [metadata])

    collection_info = vector_db.get_collection_info()    
    document_count = collection_info['document_count']
    print(f"Finished processing {len(chunks)} chunks and new collection count is {document_count}")


def index_markdown_from_directory(dir: str, vector_db: ChromaVectorDB, embedding_model: EmbeddingModel):    
    print(f"Found index_markdown_from_directory files from directory {dir}")
    if os.path.isdir(dir):                    
        md_files = Path(dir).glob("*.md")
        count = 0
        
        for count, md_file in enumerate(md_files, 1):
            #print(f"index_markdown_file {md_file}\n")
            index_markdown_file(md_file, vector_db, embedding_model)
        print(f"Finished processing {count} files from directory {dir}")
    else:
        print(f"The directory '{dir}' does not exist.")
        raise FileExistsError(f"The directory '{dir}' does not exist.")

    
def main(collection_name: str, persist_directory: str, file: str, dir: str):
    print(f"========== main: {file}, {dir} ==========")
    load_dotenv()
    

    print(f"Creating vector database: {collection_name}, {persist_directory}")
    vector_db = ChromaVectorDB.create(collection_name, persist_directory)
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    print(f"embedding_model_name: {embedding_model_name}")
    embedding_model = EmbeddingModel.from_name(embedding_model_name)
    if file:
        index_markdown_file(file, vector_db, embedding_model)
    elif dir:
        index_markdown_from_directory(dir, vector_db, embedding_model)
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
    if not args.collection_name and not args.persist_directory:
        print("Please provide --collection_name and --persist_directory arguments.")
        raise ValueError("Please provide --collection_name and --persist_directory arguments.")
    elif not args.file and not args.dir:
        print("Please provide --file or --dir arguments.")
        raise ValueError("Please provide --file or --dir arguments.")
    else:
        main(args.collection_name, args.persist_directory, args.file, args.dir)
