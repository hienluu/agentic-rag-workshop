"""
Shared data models for the RAG application.
"""
from dataclasses import dataclass
from langchain_core.documents import Document

@dataclass
class CrawlPageResult:
    """Result of crawling a web page."""
    page_url: str
    page_title: str
    page_content: str


@dataclass
class MarkdownFile:
    """A markdown file."""
    page_info: CrawlPageResult
    chunks: list[Document]