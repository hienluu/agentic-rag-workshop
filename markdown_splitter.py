import argparse
import os
import json
import asyncio
from pathlib import Path
from dataclasses import asdict

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from models import CrawlPageResult, MarkdownFile


COOKIE_MESSAGE = "Your choice regarding cookies on this site"

def split_markdown(markdown: str) -> list[Document]:
    headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                    ("####", "Header 4"),
                    ("#####", "Header 5"),
                    ("######", "Header 6"),
                ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, 
            strip_headers=True)

    chunks = splitter.split_text(markdown)
    return chunks
    
def split_markdown_file(md_file: str) -> MarkdownFile:
    print(f"Indexing markdown file: {md_file}")
    # Placeholder for actual indexing logic
    # This could involve reading the file, chunking content, and storing in a vector store
    # Load the object from the pickle file
    
    crawlPageResult = None
    with open(md_file, "rb") as f:
        data_dict = json.load(f)
        crawlPageResult = CrawlPageResult(**data_dict)
    
    if not crawlPageResult:
        print(f"Failed to load CrawlPageResult from {md_file}")
        return

        
    chunks = split_markdown(crawlPageResult.page_content)

    # remove the first chunk if it contains the cookie message
    if len(chunks) > 0:
        if COOKIE_MESSAGE in chunks[0].page_content:
            chunks = chunks[1:]
    
    return MarkdownFile(page_info=crawlPageResult, chunks=chunks)

def process_markdown_file(md_file: str):
    print(f"Indexing markdown file: {md_file}")
    # Placeholder for actual indexing logic
    # This could involve reading the file, chunking content, and storing in a vector store
    # Load the object from the pickle file
    
    markdown_file = split_markdown_file(md_file)

    crawlPageResult = markdown_file.page_info
    
    
    if not crawlPageResult:
        print(f"Failed to load CrawlPageResult from {md_file}")
        return
    
    print(f"Loaded CrawlPageResult:")
    print(f"url: {crawlPageResult.page_url}")
    print(f"title: {crawlPageResult.page_title}")
        
    chunks = markdown_file.chunks

    # remove the first chunk if it contains the cookie message
    if len(chunks) > 0:
        if COOKIE_MESSAGE in chunks[0].page_content:
            chunks = chunks[1:]
            
    print(f"Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks): 
        print(f"\n>>>> Chunk {i+1}:\n")
        if len(chunk.metadata.keys()) > 0:
            # add the header back to the begginng of the chunk
            last_key = list(chunk.metadata.keys())[-1]            
            header_text = chunk.metadata[last_key]            
            chunk.page_content = f"{header_text}\n\n{chunk.page_content}"

           
        print(chunk.page_content)
        print(f">>>> metadata\n")
            
        headers = []
        keys = list(chunk.metadata.keys())
        for key in keys[:-1]:            
                headers.append(chunk.metadata[key])
        print(f"headers: {" >> ".join(headers)}\n")
    

def main(md_file: str, input_dir: str):
    if md_file:
        if os.path.isfile(md_file):
            print(f"Indexing single file: {md_file}")
            process_markdown_file(md_file)
        else:
            print(f"The file '{md_file}' does not exist.")
            raise FileExistsError(f"The file '{md_file}' does not exist.")             
    elif input_dir:
        if os.path.isdir(input_dir):            
            print(f"Indexing all markdown files in directory: {input_dir}")
            md_files = Path(input_dir).glob("*.md")
            print(f"Found markdown files: {list(md_files)}")
            for md_file in md_files:
                print(f"Indexing file: {md_file}")
                process_markdown_file(str(md_file))            
        else:
            print(f"The directory '{input_dir}' does not exist.")
            raise FileExistsError(f"The directory '{input_dir}' does not exist.")
       
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split content from crawled markdown files."
    )
    parser.add_argument("--file", "-f", help="Markdown file to index", required=False)
    parser.add_argument( "--dir", "-d", help="Directory containing the markdown files", required=False)
    args = parser.parse_args()
    print(args)
    
    if not args.file and not args.dir:
        print("Please provide either --file or --dir argument to index the markdown file(s)")        
    else:    
        main(args.file, args.dir)