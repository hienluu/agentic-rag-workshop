import argparse
import os
import json
import asyncio
from pathlib import Path
from dataclasses import asdict

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from models import CrawlPageResult


class WebCrawlerService:
    def __init__(self):
        self.browser_config = BrowserConfig(headless=True, text_mode=True)
        self.crawler = None
        
        self.raw_md_generator = DefaultMarkdownGenerator(
            content_source="cleaned_html",
            options={
                "ignore_links": True,
                "ignore_images": True,
                "escape_html": False,               
            },          
        )
        
        self.run_config = CrawlerRunConfig(      
            excluded_tags=['form', 'header', 'footer', 'nav'],
            markdown_generator=self.raw_md_generator,
            cache_mode=CacheMode.BYPASS,
        )
        
        print("==========WebCrawlerService initialized==============")        
        print(self.raw_md_generator.content_source)
        print("==========raw_md_generator.options==============")
        print(self.raw_md_generator.options)
        print("==========run_config==============")
        print(f"run_config.target_elements: {self.run_config.target_elements}")
        print(f"run_config.excluded_tags: {self.run_config.excluded_tags}")
        print(f"run_config.css_selector: {self.run_config.css_selector}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.crawler = AsyncWebCrawler(config=self.browser_config)
        await self.crawler.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    async def crawl_page(self, url: str) -> CrawlPageResult:
        """Crawl a single page and return the result"""
        print(f"==========crawl_page: {url}==============")
        
        if not self.crawler:
            raise RuntimeError("WebCrawlerService must be used as an async context manager")
        
        result = await self.crawler.arun(
            url=url,
            config=self.run_config
        )
        
        print(f"==========result.success: {result.success} ==============")
        print(f"==========result.status_code: {result.status_code} ==============")
        
        if result.success:
            title = result.metadata.get("title", "No title found")
            print(f"The title of '{url}' is: {title}")
        else:
            print(f"Failed to crawl '{url}'.")
            title = "Failed to crawl"
        
        return CrawlPageResult(page_url=url, 
                            page_title=title, 
                            page_content=str(result.markdown))


async def crawl_page(url: str) -> CrawlPageResult:
    """Legacy function for backward compatibility"""
    async with WebCrawlerService() as crawler_service:
        return await crawler_service.crawl_page(url)

async def crawl_page_save_to_file_async(crawler_service: WebCrawlerService, url: str, output_dir: str):
    """Async version that uses the shared crawler service"""
    print("==========crawl_page_save_to_file_async==============")
    path_object = Path(url)
    filename = path_object.name    

    # crawl the page and get markdown content
    crawlPageResult = await crawler_service.crawl_page(url)
   
    # save markdown content to a file
    file_path = f"{output_dir}/{filename}.md"
    print(f"Saving markdown content to {file_path}")
    
    # TODO: make sure the file_path doesn't already exist
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(asdict(crawlPageResult), f, indent=4)
    
    print(f"crawlPageResult content saved to {file_path}")

def crawl_page_save_to_file(url: str, output_dir: str):
    """Legacy synchronous function for backward compatibility"""
    print("==========crawl_page_save_to_file==============")
    path_object = Path(url)
    filename = path_object.name    

    # crawl the page and get markdown content
    crawlPageResult = asyncio.run(crawl_page(url))
   
    # save markdown content to a file
    file_path = f"{output_dir}/{filename}.md"
    print(f"Saving markdown content to {file_path}")
    
    # TODO: make sure the file_path doesn't already exist
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(asdict(crawlPageResult), f, indent=4)
    
    print(f"crawlPageResult content saved to {file_path}")

async def main_async(url_file: str, output_dir: str):
    """Async version of main that efficiently reuses the crawler"""
    print(f"Reading URLs from {url_file} and writing output to {output_dir}")
    
    if os.path.isfile(url_file):
        print(f"The file '{url_file}' exists.")
    else:
        print(f"The file '{url_file}' does not exist.")
        raise FileExistsError(f"The file '{url_file}' does not exist.")
    
    if os.path.isdir(output_dir):
        print(f"The directory '{output_dir}' exists.")
    else:
        print(f"The directory '{output_dir}' does not exist.")
        raise FileExistsError(f"The directory '{output_dir}' does not exist.")
    
    url_list = []
    try:
        with open(url_file, 'r') as f:
            url_list = json.load(f)        
    except json.JSONDecodeError:
        print(f"Error: The file '{url_file}' is not a valid JSON file.")    

    print(f"Found a total of {len(url_list)} urls to process")
    
    # Use the WebCrawlerService as a context manager to reuse the crawler instance
    async with WebCrawlerService() as crawler_service:
        for url in url_list:
            print(f"Crawling url: {url}")
            await crawl_page_save_to_file_async(crawler_service, url, output_dir)
    
    print(f"Processed a total {len(url_list)} urls")
    print("====== done processing =============")
    
def main(url_file: str, output_dir: str):
    """Synchronous main function that calls the async version"""
    asyncio.run(main_async(url_file, output_dir))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl the urls from the url list file,and get its markdown representation, and optionally save them to a directory"
    )
    parser.add_argument("--file", "-f", help="File contains a list of URLs in json format to crawl.", required=True)
    parser.add_argument( "--output", "-o", default="output.md", help="output directory to save the markdown files", 
                        required=True)
    args = parser.parse_args()
    print(args)
    main(args.file, args.output)
  