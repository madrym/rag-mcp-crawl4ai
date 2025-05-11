import subprocess
import sys
import os
import pytest
from unittest import mock

CLI_PATH = os.path.join(os.path.dirname(__file__), '../src/rag_cli.py')

# Import the refactored logic for in-process testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import rag_cli

def run_cli(args, env=None):
    """Helper to run the CLI and return (exit_code, stdout, stderr)"""
    result = subprocess.run(
        [sys.executable, CLI_PATH] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or os.environ.copy()
    )
    return result.returncode, result.stdout, result.stderr


def test_cli_no_args_shows_help():
    code, out, err = run_cli([])
    assert code == 1
    assert "RAG CLI" in err or "RAG CLI" in out
    assert "rag-query" in err or "rag-query" in out
    assert "crawl-website" in err or "crawl-website" in out
    assert "crawl-github" in err or "crawl-github" in out


def test_cli_rag_query_help():
    code, out, err = run_cli(["rag-query", "--help"])
    assert code == 0
    assert "--query" in out
    assert "--openai-endpoint" in out
    assert "--openai-model" in out
    assert "--openai-api-key" in out


def test_cli_crawl_website_help():
    code, out, err = run_cli(["crawl-website", "--help"])
    assert code == 0
    assert "--url" in out
    assert "--max-depth" in out
    assert "--chunk-size" in out


def test_cli_crawl_github_help():
    code, out, err = run_cli(["crawl-github", "--help"])
    assert code == 0
    assert "--repo-url" in out
    assert "--branch" in out
    assert "--max-depth" in out
    assert "--chunk-size" in out
    assert "--github-token" in out


def test_cli_rag_query_missing_required():
    code, out, err = run_cli(["rag-query"])
    assert code == 2  # argparse returns 2 for argument errors
    assert "--query" in err
    assert "--openai-endpoint" in err
    assert "--openai-model" in err
    assert "--openai-api-key" in err


def make_rag_args(**kwargs):
    class Args:
        pass
    args = Args()
    args.command = "rag-query"
    args.query = kwargs.get("query", "Test question?")
    args.source = kwargs.get("source", None)
    args.match_count = kwargs.get("match_count", 5)
    args.openai_endpoint = kwargs.get("openai_endpoint", "https://fake.endpoint")
    args.openai_model = kwargs.get("openai_model", "gpt-4")
    args.openai_api_key = kwargs.get("openai_api_key", "sk-test")
    return args

@mock.patch("rag_cli.search_documents")
@mock.patch("rag_cli.call_openai_completion")
def test_rag_query_success(mock_call_openai, mock_search_documents):
    mock_search_documents.return_value = [
        {"url": "http://example.com/1", "content": "Chunk 1 content.", "similarity": 0.95},
        {"url": "http://example.com/2", "content": "Chunk 2 content.", "similarity": 0.93},
    ]
    mock_call_openai.return_value = "This is a generated answer."
    args = make_rag_args(query="What is test?")
    with mock.patch("builtins.print") as mock_print:
        code = rag_cli.run_rag_query(args)
    output = "\n".join(str(c[0]) for c in mock_print.call_args_list)
    assert code == 0
    assert "[INFO] Running RAG query" in output
    assert "--- Chunk 1 ---" in output
    assert "Chunk 1 content." in output
    assert "--- Chunk 2 ---" in output
    assert "Chunk 2 content." in output
    assert "[ANSWER]" in output
    assert "This is a generated answer." in output

@mock.patch("rag_cli.search_documents")
def test_rag_query_no_results(mock_search_documents):
    mock_search_documents.return_value = []
    args = make_rag_args(query="No results?")
    with mock.patch("builtins.print") as mock_print:
        code = rag_cli.run_rag_query(args)
    output = "\n".join(str(c[0]) for c in mock_print.call_args_list)
    assert code == 0
    assert "[WARN] No relevant chunks found in the database." in output

@mock.patch("rag_cli.search_documents")
@mock.patch("rag_cli.call_openai_completion")
def test_rag_query_openai_error(mock_call_openai, mock_search_documents):
    mock_search_documents.return_value = [
        {"url": "http://example.com/1", "content": "Chunk 1 content.", "similarity": 0.95}
    ]
    mock_call_openai.side_effect = Exception("OpenAI error!")
    args = make_rag_args(query="What is test?")
    with mock.patch("builtins.print") as mock_print:
        code = rag_cli.run_rag_query(args)
    output = "\n".join(str(c[0]) for c in mock_print.call_args_list)
    assert code == 1
    assert "[ERROR] RAG query failed" in output

def make_crawl_github_args(**kwargs):
    class Args:
        pass
    args = Args()
    args.command = "crawl-github"
    args.repo_url = kwargs.get("repo_url", "https://github.com/user/repo")
    args.branch = kwargs.get("branch", "main")
    args.max_depth = kwargs.get("max_depth", 2)
    args.chunk_size = kwargs.get("chunk_size", 5000)
    args.github_token = kwargs.get("github_token", None)
    return args

@mock.patch("rag_cli.add_documents_batch")
@mock.patch("rag_cli.asyncio.run")
def test_crawl_github_api_success(mock_asyncio_run, mock_add_documents):
    # Simulate API returns two files with content
    all_files = ["file1.md", "file2.md"]
    file_contents = ["# File 1\nContent", "# File 2\nContent"]
    mock_asyncio_run.return_value = (all_files, file_contents)
    args = make_crawl_github_args(repo_url="https://github.com/user/repo", github_token="tok")
    with mock.patch("builtins.print") as mock_print:
        code = rag_cli.run_crawl_github(args)
    output = "\n".join(str(c[0]) for c in mock_print.call_args_list)
    assert code == 0
    assert "[INFO] Crawling GitHub repo user/repo" in output
    assert "Files crawled: 2" in output
    assert mock_add_documents.called

@mock.patch("rag_cli.add_documents_batch")
@mock.patch("requests.get")
def test_crawl_github_raw_success(mock_requests_get, mock_add_documents):
    requested_urls = []
    def fake_get(url, headers=None):
        requested_urls.append(url)
        print(f"DEBUG: requests.get called with url={url}")
        class Resp:
            def __init__(self, text, status_code=200):
                self.text = text
                self.status_code = status_code
                self.headers = {"content-type": "text/plain"}
            def json(self):
                if "/git/trees/" in url or "/trees/" in url:
                    return {"tree": [
                        {"path": "file1.md", "type": "blob"},
                        {"path": "file2.md", "type": "blob"}
                    ]}
                return {}
        if "/git/trees/" in url or "/trees/" in url:
            return Resp("", 200)
        if "raw.githubusercontent.com" in url:
            if url.endswith("file1.md"):
                return Resp("# File 1\nContent")
            if url.endswith("file2.md"):
                return Resp("# File 2\nContent")
        return Resp("", 404)
    mock_requests_get.side_effect = fake_get
    os.environ.pop('GITHUB_TOKEN', None)
    args = make_crawl_github_args(repo_url="https://github.com/user/repo", github_token=None)
    with mock.patch("builtins.print") as mock_print:
        code = rag_cli.run_crawl_github(args)
    output = "\n".join(str(c[0]) for c in mock_print.call_args_list)
    print("DEBUG: requested_urls:", requested_urls)
    print("DEBUG: add_documents_batch call args:", mock_add_documents.call_args_list)
    print("DEBUG: code returned:", code)
    print("DEBUG: captured output:\n", output)
    assert code == 0
    assert "[INFO] Crawling GitHub repo user/repo" in output
    assert "Files crawled: 2" in output
    assert mock_add_documents.called

@mock.patch("rag_cli.add_documents_batch")
@mock.patch("rag_cli.asyncio.run")
def test_crawl_github_api_error(mock_asyncio_run, mock_add_documents):
    mock_asyncio_run.side_effect = Exception("API error!")
    args = make_crawl_github_args(repo_url="https://github.com/user/repo", github_token="tok")
    with mock.patch("builtins.print") as mock_print:
        code = rag_cli.run_crawl_github(args)
    output = "\n".join(str(c[0]) for c in mock_print.call_args_list)
    assert code == 1
    assert "[ERROR] GitHub crawl failed" in output
    assert not mock_add_documents.called

@mock.patch("rag_cli.add_documents_batch")
@mock.patch("requests.get")
def test_crawl_github_raw_error(mock_requests_get, mock_add_documents):
    def fake_get(url, headers=None):
        class Resp:
            status_code = 404
            text = ""
            headers = {"content-type": "text/plain"}
            def json(self):
                return {"tree": []}
        return Resp()
    mock_requests_get.side_effect = fake_get
    args = make_crawl_github_args(repo_url="https://github.com/user/repo", github_token=None)
    with mock.patch("builtins.print") as mock_print:
        code = rag_cli.run_crawl_github(args)
    output = "\n".join(str(c[0]) for c in mock_print.call_args_list)
    assert code == 1
    assert "[WARN] No files found to crawl." in output
    assert not mock_add_documents.called 