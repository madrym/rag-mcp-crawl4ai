import argparse
import sys
import os
from typing import List, Set, Dict
import requests
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup
import time
import aiohttp
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Import utilities from utils.py
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils import get_db_connection, search_documents, close_db_connection, add_documents_batch
from utils import create_embedding  # for testability
from utils import smart_chunk_markdown, extract_section_info
import openai


def call_openai_completion(endpoint, api_key, model, context_chunks: List[str], user_query: str) -> str:
    # Set up OpenAI client for custom endpoint
    openai.api_key = api_key
    if endpoint:
        openai.base_url = endpoint.rstrip("/")
        print(f"[DEBUG] Using custom OpenAI endpoint: {openai.base_url}")
    else:
        print(f"[DEBUG] Using OpenAI package default endpoint")
    # Compose the prompt/context
    context = "\n\n".join(context_chunks)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    ]
    print(f"[DEBUG] Model: {model}")
    print(f"[DEBUG] API Key starts with: {api_key[:6]}... (length: {len(api_key)})")
    print(f"[DEBUG] Prompt preview: {messages[-1]['content'][:200]}{'...' if len(messages[-1]['content']) > 200 else ''}")
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] OpenAI completion failed: {e}"


def run_rag_query(args):
    print(f"[INFO] Running RAG query: '{args.query}'")
    try:
        filter_metadata = {"source": args.source} if args.source else None
        results = search_documents(
            query=args.query,
            match_count=args.match_count,
            filter_metadata=filter_metadata
        )
        # Filter by min similarity
        min_sim = args.min_similarity
        filtered_results = [r for r in results if r.get('similarity', 0) >= min_sim]
        if not filtered_results:
            print(f"[WARN] No relevant chunks found in the local database with similarity >= {min_sim}.")
            print("[INFO] No local context was found for your query. The answer below is not based on any local data.")
            answer = call_openai_completion(
                endpoint=args.openai_endpoint,
                api_key=args.openai_api_key,
                model=args.openai_model,
                context_chunks=[],
                user_query=args.query
            )
            print("\n[ANSWER]\n" + answer)
            return 0
        print(f"[INFO] Top {len(filtered_results)} retrieved chunks (similarity >= {min_sim}):")
        for i, res in enumerate(filtered_results, 1):
            print(f"\n--- Chunk {i} ---\nURL: {res.get('url')}\nSimilarity: {res.get('similarity', 'N/A')}\nContent:\n{res.get('content')[:500]}{'...' if len(res.get('content', '')) > 500 else ''}")
        context_chunks = [r["content"] for r in filtered_results]
        answer = call_openai_completion(
            endpoint=args.openai_endpoint,
            api_key=args.openai_api_key,
            model=args.openai_model,
            context_chunks=context_chunks,
            user_query=args.query
        )
        print("\n[ANSWER]\n" + answer)
        return 0
    except Exception as e:
        print(f"[ERROR] RAG query failed: {e}")
        return 1
    finally:
        close_db_connection()


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts, styles, nav, footer, header, aside
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    # Get text from <main> if present, else body
    main = soup.find("main")
    text = main.get_text(separator="\n") if main else soup.body.get_text(separator="\n") if soup.body else soup.get_text(separator="\n")
    return text.strip()


def is_internal_link(link: str, base_netloc: str) -> bool:
    parsed = urlparse(link)
    return (not parsed.netloc or parsed.netloc == base_netloc) and (parsed.scheme in ("http", "https", ""))


def normalize_url(url: str) -> str:
    return urldefrag(url)[0]


def run_crawl_website(args):
    start_url = args.url
    max_depth = args.max_depth
    chunk_size = args.chunk_size
    visited: Set[str] = set()
    to_visit: Set[str] = set([normalize_url(start_url)])
    all_results: Dict[str, str] = {}
    base_netloc = urlparse(start_url).netloc
    depth = 0
    print(f"[INFO] Starting crawl at {start_url} (max_depth={max_depth})")
    try:
        while to_visit and depth < max_depth:
            print(f"[INFO] Crawling depth {depth+1} with {len(to_visit)} URLs...")
            current_level = list(to_visit)
            to_visit.clear()
            for url in current_level:
                if url in visited:
                    continue
                try:
                    resp = requests.get(url, timeout=15)
                    if resp.status_code != 200 or not resp.headers.get("content-type", "").startswith("text/html"):
                        print(f"[WARN] Skipping {url} (status {resp.status_code}, content-type {resp.headers.get('content-type')})")
                        visited.add(url)
                        continue
                    html = resp.text
                    text = extract_main_text(html)
                    all_results[url] = text
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        link = urljoin(url, a["href"])
                        link = normalize_url(link)
                        if is_internal_link(link, base_netloc) and link not in visited and link not in all_results:
                            to_visit.add(link)
                except Exception as e:
                    print(f"[ERROR] Failed to fetch {url}: {e}")
                visited.add(url)
            depth += 1
        if not all_results:
            print("[WARN] No HTML pages successfully crawled.")
            return 1
        # Chunk and store
        urls, chunk_numbers, contents, metadatas = [], [], [], []
        for url, text in all_results.items():
            chunks = smart_chunk_markdown(text, chunk_size=chunk_size)
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = base_netloc
                meta["crawl_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                metadatas.append(meta)
        url_to_full_document = dict(all_results)
        add_documents_batch(urls, chunk_numbers, contents, metadatas, url_to_full_document)
        print(f"[INFO] Crawl complete. Pages crawled: {len(all_results)}. Chunks stored: {len(contents)}.")
        return 0
    except Exception as e:
        print(f"[ERROR] Website crawl failed: {e}")
        return 1
    finally:
        close_db_connection()


def run_crawl_github(args):
    repo_url = args.repo_url.rstrip('/')
    branch = args.branch
    max_depth = args.max_depth
    chunk_size = args.chunk_size
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    exclude_exts = {'.png', '.jpg', '.jpeg', '.gif', '.exe', '.bin', '.pdf', '.zip', '.tar', '.gz'}
    exclude_names = {'README.md', 'LICENSE', 'LICENSE.txt'}
    parsed = urlparse(repo_url)
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) < 2:
        print("[ERROR] Invalid GitHub repo URL.")
        return 1
    owner, repo = path_parts[:2]
    async def fetch_github_api_files():
        headers = {'Authorization': f'token {github_token}'} if github_token else {}
        async def list_files(session, owner, repo, path, depth):
            if depth > max_depth:
                return []
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
            async with session.get(api_url, headers=headers) as resp:
                if resp.status != 200:
                    return []
                items = await resp.json()
                files = []
                for item in items:
                    name = item['name']
                    ext = '.' + name.split('.')[-1] if '.' in name else ''
                    if item['type'] == 'file':
                        if (ext in exclude_exts) or (name in exclude_names):
                            continue
                        files.append(item['path'])
                    elif item['type'] == 'dir':
                        files.extend(await list_files(session, owner, repo, item['path'], depth+1))
                return files
        async def fetch_file(session, raw_url):
            async with session.get(raw_url, headers=headers) as resp:
                if resp.status == 200:
                    raw_bytes = await resp.read()
                    try:
                        return raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        print(f"[WARN] Skipping non-UTF-8 file: {raw_url}")
                        return None
                return None
        async with aiohttp.ClientSession() as session:
            all_files = await list_files(session, owner, repo, '', 0)
            tasks = []
            for file_path in all_files:
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
                tasks.append(fetch_file(session, raw_url))
            file_contents = await asyncio.gather(*tasks)
        return all_files, file_contents
    def fetch_raw_urls():
        # Simple recursive crawl for public repos (raw URLs)
        # Try both /git/trees/ and /trees/ endpoints for robustness
        import requests
        tree_urls = [
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
            f"https://api.github.com/repos/{owner}/{repo}/trees/{branch}?recursive=1"
        ]
        headers = {'Authorization': f'token {github_token}'} if github_token else {}
        tree = []
        for tree_url in tree_urls:
            resp = requests.get(tree_url, headers=headers)
            print(f"DEBUG: Trying tree_url={tree_url}, status={resp.status_code}")
            if resp.status_code == 200:
                resp_json = resp.json()
                print(f"DEBUG: tree_url response json={resp_json}")
                tree = resp_json.get('tree', [])
                if tree:
                    break
        if not tree:
            print(f"[ERROR] Could not fetch repo tree: {resp.status_code}")
            return [], []
        files = [item['path'] for item in tree if item['type'] == 'blob']
        filtered_files = []
        for file_path in files:
            name = file_path.split('/')[-1]
            ext = '.' + name.split('.')[-1] if '.' in name else ''
            if (ext in exclude_exts) or (name in exclude_names):
                continue
            filtered_files.append(file_path)
        file_contents = []
        for file_path in filtered_files:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
            try:
                resp = requests.get(raw_url)
                if resp.status_code == 200:
                    try:
                        text = resp.content.decode("utf-8")
                        file_contents.append(text)
                    except UnicodeDecodeError:
                        print(f"[WARN] Skipping non-UTF-8 file: {raw_url}")
                        file_contents.append(None)
                else:
                    file_contents.append(None)
            except Exception:
                file_contents.append(None)
        return filtered_files, file_contents
    print(f"[INFO] Crawling GitHub repo {owner}/{repo} (branch={branch}, max_depth={max_depth})")
    try:
        if github_token:
            all_files, file_contents = asyncio.run(fetch_github_api_files())
        else:
            all_files, file_contents = fetch_raw_urls()
        if not all_files:
            print("[WARN] No files found to crawl.")
            return 1
        print(f"DEBUG: filtered_files={len(all_files)}, file_contents={len(file_contents)}")
        urls, chunk_numbers, contents, metadatas = [], [], [], []
        url_to_full_document = {}
        for file_path, text in zip(all_files, file_contents):
            if not text:
                continue
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
            url_to_full_document[raw_url] = text
            chunks = smart_chunk_markdown(text, chunk_size=chunk_size)
            print(f"DEBUG: file_path={file_path}, content_len={len(text)}, num_chunks={len(chunks)}")
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
        if not urls:
            print("[WARN] No valid file contents to store.")
            return 1
        add_documents_batch(urls, chunk_numbers, contents, metadatas, url_to_full_document)
        print(f"[INFO] GitHub crawl complete. Files crawled: {len(all_files)}. Chunks stored: {len(contents)}.")
        return 0
    except Exception as e:
        print(f"[ERROR] GitHub crawl failed: {e}")
        return 1
    finally:
        close_db_connection()


def build_parser():
    parser = argparse.ArgumentParser(
        description="RAG CLI: Retrieve, crawl websites, and crawl GitHub repos with OpenAI-compatible and pgvector support."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run. Use '<command> --help' for details.")

    # rag-query subcommand
    rag_parser = subparsers.add_parser(
        "rag-query",
        help="Retrieve and generate answers using RAG (Retrieval Augmented Generation)."
    )
    rag_parser.add_argument("--query", required=True, help="The user query/question to answer.")
    rag_parser.add_argument("--source", required=False, help="Optional: Filter results by source/domain.")
    rag_parser.add_argument("--match-count", type=int, default=5, help="Number of chunks to retrieve (default: 5).")
    rag_parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.6,
        help="Minimum similarity score for a chunk to be used as context (default: 0.6)."
    )
    rag_parser.add_argument(
        "--openai-endpoint",
        required=False,
        default=os.environ.get("OPENAI_ENDPOINT"),
        help="Optional: OpenAI-compatible API endpoint URL. Can also be set via OPENAI_ENDPOINT env var. Defaults to OpenAI package default if not set."
    )
    rag_parser.add_argument(
        "--openai-model",
        required=False,
        default=os.environ.get("OPENAI_MODEL"),
        help="Model name for completion (e.g., 'gpt-4'). Can also be set via OPENAI_MODEL env var."
    )
    rag_parser.add_argument(
        "--openai-api-key",
        required=False,
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key for the OpenAI-compatible endpoint. Can also be set via OPENAI_API_KEY env var."
    )

    # crawl-website subcommand
    crawl_web_parser = subparsers.add_parser(
        "crawl-website",
        help="Crawl a website and store its content in the database."
    )
    crawl_web_parser.add_argument("--url", required=True, help="The website URL to crawl.")
    crawl_web_parser.add_argument("--max-depth", type=int, default=3, help="Recursion depth for internal links (default: 3).")
    crawl_web_parser.add_argument("--chunk-size", type=int, default=5000, help="Max chunk size for content splitting (default: 5000).")

    # crawl-github subcommand
    crawl_gh_parser = subparsers.add_parser(
        "crawl-github",
        help="Crawl a GitHub repository and store its content in the database."
    )
    crawl_gh_parser.add_argument("--repo-url", required=True, help="The GitHub repository URL to crawl.")
    crawl_gh_parser.add_argument("--branch", default="main", help="Branch to crawl (default: 'main').")
    crawl_gh_parser.add_argument("--max-depth", type=int, default=3, help="Directory recursion depth (default: 3).")
    crawl_gh_parser.add_argument("--chunk-size", type=int, default=5000, help="Max chunk size for content splitting (default: 5000).")
    crawl_gh_parser.add_argument("--github-token", required=False, help="GitHub token for private repos or higher rate limits.")

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    if args.command == "rag-query":
        # Validate required args (env or CLI)
        missing = []
        if not args.openai_model:
            missing.append("--openai-model or OPENAI_MODEL")
        if not args.openai_api_key:
            missing.append("--openai-api-key or OPENAI_API_KEY")
        if missing:
            print(f"[ERROR] Missing required argument(s): {', '.join(missing)}")
            sys.exit(2)
        sys.exit(run_rag_query(args))
    elif args.command == "crawl-website":
        sys.exit(run_crawl_website(args))
    elif args.command == "crawl-github":
        sys.exit(run_crawl_github(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 