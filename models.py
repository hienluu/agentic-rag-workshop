"""
Shared data models for the RAG application.
"""
from dataclasses import dataclass


@dataclass
class CrawlPageResult:
    """Result of crawling a web page."""
    page_url: str
    page_title: str
    page_content: str

