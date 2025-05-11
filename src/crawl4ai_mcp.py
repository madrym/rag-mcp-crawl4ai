"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urlparse
import requests
import asyncio
import json
import os
import re
import aiohttp
import time

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from utils import get_db_connection, close_db_connection, add_documents_batch, search_documents
from utils import smart_chunk_markdown, extract_section_info

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler

# Add a global server_ready flag
server_ready = False

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler
    """
    global server_ready
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    server_ready = True  # Set ready flag after initialization
    try:
        yield Crawl4AIContext(
            crawler=crawler
        )
    finally:
        server_ready = False
        print("Shutting down: closing crawler...")
        await crawler.__aexit__(None, None, None)
        print("Shutting down: closing database connection...")
        close_db_connection()
        print("Shutdown complete.")

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
    
    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        
        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
            
            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}
            
            # Add to Supabase
            add_documents_batch(urls, chunk_numbers, contents, metadatas, url_to_full_document)
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.
    
    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is chunked and stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 1000)
    
    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        
        crawl_results = []
        crawl_type = "webpage"
        
        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            
            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = urlparse(source_url).netloc
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                chunk_count += 1
        
        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc['url']] = doc['markdown']
        
        # Add to Supabase
        # IMPORTANT: Adjust this batch size for more speed if you want! Just don't overwhelm your system or the embedding API ;)
        batch_size = 50
        add_documents_batch(urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size=batch_size)
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.
    
    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources based on unique source metadata values.
    
    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string with the list of available sources
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        
        # Use a direct query with the database
        # This could be more efficient with a direct Postgres query but
        # I don't want to require users to set a DB_URL environment variable as well
        result = search_documents(
            query="SELECT DISTINCT metadata->>source FROM crawled_pages WHERE metadata->>source IS NOT NULL",
            match_count=None
        )
            
        # Use a set to efficiently track unique sources
        unique_sources = set()
        
        # Extract the source values from the result using a set for uniqueness
        if result:
            for item in result:
                source = item.get('source')
                if source:
                    unique_sources.add(source)
        
        # Convert set to sorted list for consistent output
        sources = sorted(list(unique_sources))
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    
    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.

    Use the tool to get source domains if the user is asking to use a specific tool or framework.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    global server_ready
    if not server_ready:
        return json.dumps({"success": False, "error": "Server not ready. Please try again in a few seconds."})
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}
        
        # Perform the search
        results = search_documents(
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def crawl_github_repo(
    ctx: Context,
    repo_url: str,
    max_depth: int = 3,
    chunk_size: int = 5000,
    github_token: Optional[str] = None,
    branch: str = "main",
    exclude_extensions: Optional[List[str]] = None,
    exclude_filenames: Optional[List[str]] = None
) -> str:
    """
    Crawl a GitHub repository and store its content in Supabase.
    - If github_token is provided, uses the GitHub API (works for private and public repos, more efficient).
    - If no github_token is provided, uses Crawl4AI's browser-based crawling (works for public repos, no API limits).
    - In both cases, supports filtering by branch (API only), excluding files by extension or name, and chunking content for RAG.
    - All chunks use extract_section_info for metadata consistency.
    Args:
        ctx: The MCP server provided context
        repo_url: URL of the GitHub repository (e.g., https://github.com/user/repo)
        max_depth: Maximum directory recursion depth (default: 3)
        chunk_size: Maximum size of each content chunk in characters (default: 5000)
        github_token: Optional GitHub token for private repos or higher rate limits
        branch: Branch to crawl (default: "main")
        exclude_extensions: List of file extensions to exclude (e.g., [".png", ".exe"])
        exclude_filenames: List of file names to exclude (e.g., ["README.md", "LICENSE"])
    Returns:
        JSON string with crawl summary and storage information
    """
    crawler = ctx.request_context.lifespan_context.crawler
    exclude_exts = set(exclude_extensions or [])
    exclude_names = set(exclude_filenames or [])
    github_token = os.getenv("GITHUB_TOKEN")
    repo_url = repo_url.rstrip('/')
    parsed = urlparse(repo_url)
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) < 2:
        return json.dumps({"success": False, "error": "Invalid GitHub repo URL"}, indent=2)
    owner, repo = path_parts[:2]
    
    if github_token:
        # --- API-based approach (private or public repos) ---
        headers = {'Authorization': f'token {github_token}'}
        async def fetch_github_file(session, raw_url, headers):
            try:
                async with session.get(raw_url, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    elif resp.status == 403:
                        reset = resp.headers.get("X-RateLimit-Reset")
                        if reset:
                            wait_time = int(reset) - int(time.time())
                            return f"RATE_LIMIT:{wait_time}"
                        return None
                    return None
            except Exception as e:
                return f"ERROR:{str(e)}"
        async def list_github_files(session, owner, repo, path, headers, depth, max_depth, branch, exclude_exts, exclude_names):
            if depth > max_depth:
                return []
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
            try:
                async with session.get(api_url, headers=headers) as resp:
                    if resp.status == 403:
                        reset = resp.headers.get("X-RateLimit-Reset")
                        wait_time = int(reset) - int(time.time()) if reset else 60
                        raise Exception(f"GitHub API rate limit exceeded. Try again in {wait_time} seconds.")
                    if resp.status != 200:
                        print(f"Error listing files at {api_url}: HTTP {resp.status}")
                        return []
                    try:
                        items = await resp.json()
                    except Exception as e:
                        print(f"JSON decode error at {api_url}: {e}")
                        return []
                    files = []
                    for item in items:
                        name = item['name']
                        ext = '.' + name.split('.')[-1] if '.' in name else ''
                        if item['type'] == 'file':
                            if (ext in exclude_exts) or (name in exclude_names):
                                continue
                            files.append(item['path'])
                        elif item['type'] == 'dir':
                            files.extend(
                                await list_github_files(
                                    session, owner, repo, item['path'], headers, depth+1, max_depth, branch, exclude_exts, exclude_names
                                )
                            )
                    return files
            except Exception as e:
                print(f"Error listing files at {api_url}: {e}")
                return []
        async with aiohttp.ClientSession() as session:
            all_files = await list_github_files(
                session, owner, repo, "", headers, 0, max_depth, branch, exclude_exts, exclude_names
            )
            tasks = []
            for file_path in all_files:
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
                tasks.append(fetch_github_file(session, raw_url, headers))
            file_contents = await asyncio.gather(*tasks)
        urls, chunk_numbers, contents, metadatas = [], [], [], []
        url_to_full_document = {}
        for file_path, text in zip(all_files, file_contents):
            if not text or text.startswith("RATE_LIMIT:") or text.startswith("ERROR:"):
                if text and text.startswith("RATE_LIMIT:"):
                    print(f"Rate limit hit for {file_path}: wait {text.split(':')[1]} seconds")
                elif text and text.startswith("ERROR:"):
                    print(f"Error fetching {file_path}: {text}")
                continue
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
            url_to_full_document[raw_url] = text
            chunks = smart_chunk_markdown(text, chunk_size=chunk_size)
            for i, chunk in enumerate(chunks):
                meta = extract_section_info(chunk)
                meta.update({
                    "chunk_index": i,
                    "url": raw_url,
                    "source": f"github.com/{owner}/{repo}",
                    "file_path": file_path,
                    "repo": f"{owner}/{repo}",
                    "branch": branch
                })
                urls.append(raw_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                metadatas.append(meta)
        add_documents_batch(urls, chunk_numbers, contents, metadatas, url_to_full_document)
        return json.dumps({
            "success": True,
            "repo": f"{owner}/{repo}",
            "branch": branch,
            "files_crawled": len(all_files),
            "chunks_stored": len(contents),
            "method": "api"
        }, indent=2)
    else:
        # --- Browser-based Crawl4AI approach (public repos) ---
        allowed_prefix = repo_url
        def is_allowed_link(url):
            return url.startswith(allowed_prefix)
        async def crawl_recursive_github(crawler, start_urls, max_depth, allowed_prefix, exclude_exts, exclude_names):
            visited = set()
            current_urls = set(start_urls)
            results_all = []
            for depth in range(max_depth):
                urls_to_crawl = [u for u in current_urls if u not in visited]
                if not urls_to_crawl:
                    break
                try:
                    crawl_results = await crawler.arun_many(urls=urls_to_crawl)
                except Exception as e:
                    print(f"Error crawling URLs at depth {depth}: {e}")
                    continue
                next_level_urls = set()
                for result in crawl_results:
                    visited.add(result.url)
                    if not getattr(result, 'success', False):
                        print(f"Failed to crawl {getattr(result, 'url', 'unknown')}: {getattr(result, 'error_message', 'Unknown error')}")
                        continue
                    if result.success and result.markdown:
                        # Exclude by extension or filename (heuristic: check url path)
                        path = urlparse(result.url).path
                        name = path.split('/')[-1]
                        ext = '.' + name.split('.')[-1] if '.' in name else ''
                        if (ext in exclude_exts) or (name in exclude_names):
                            continue
                        results_all.append({'url': result.url, 'markdown': result.markdown})
                        for link in result.links.get("internal", []):
                            href = link.get("href")
                            if href and href.startswith(allowed_prefix) and href not in visited:
                                next_level_urls.add(href)
                current_urls = next_level_urls
            return results_all
        crawl_results = await crawl_recursive_github(
            crawler, [repo_url], max_depth, allowed_prefix, exclude_exts, exclude_names
        )
        urls, chunk_numbers, contents, metadatas = [], [], [], []
        url_to_full_document = {}
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            url_to_full_document[source_url] = md
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            for i, chunk in enumerate(chunks):
                meta = extract_section_info(chunk)
                meta.update({
                    "chunk_index": i,
                    "url": source_url,
                    "source": urlparse(source_url).netloc,
                    "repo_root": repo_url
                })
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                metadatas.append(meta)
        add_documents_batch(urls, chunk_numbers, contents, metadatas, url_to_full_document)
        return json.dumps({
            "success": True,
            "repo_url": repo_url,
            "pages_crawled": len(crawl_results),
            "chunks_stored": len(contents),
            "method": "crawl4ai"
        }, indent=2)

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    try:
        if transport == 'sse':
            await mcp.run_sse_async()
        else:
            await mcp.run_stdio_async()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Received shutdown signal (KeyboardInterrupt or CancelledError). Cleaning up...")
    finally:
        print("Final cleanup (if needed) complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Force quit received. Exiting now.")