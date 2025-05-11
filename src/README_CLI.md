# RAG CLI Tool

A command-line tool for Retrieval-Augmented Generation (RAG), website crawling, and GitHub repo crawling, using OpenAI-compatible endpoints and a pgvector-backed Postgres database.

## **Features**

- **RAG Query:** Retrieve relevant context from your database and generate answers using an OpenAI-compatible LLM.
- **Website Crawling:** Recursively crawl a website, chunk and embed its content, and store it in your database.
- **GitHub Repo Crawling:** Crawl a GitHub repository (public or private), chunk and embed code/docs, and store in your database.

---

## **Installation**

1. **Clone the repository** and install dependencies:
   ```sh
   git clone <your-repo-url>
   cd mcp-crawl4ai-rag
   pip install -r requirements.txt
   ```

2. **Set up your environment variables** (for DB and OpenAI credentials).  
   See `.env.example` or your existing `.env` for details.

---

## **Usage**

Run the CLI from the `src/` directory (or with the full path):

```sh
python src/rag_cli.py <subcommand> [options]
```

### **Subcommands**

#### 1. RAG Query

Retrieve and generate answers using RAG.

```sh
python src/rag_cli.py rag-query \
  --query "What is X?" \
  --openai-endpoint "https://api.openai.com" \
  --openai-model "gpt-4" \
  --openai-api-key "<your-key>" \
  [--source "example.com"] \
  [--match-count 5]
```

- `--query` (required): The user's question.
- `--openai-endpoint` (required): OpenAI-compatible API endpoint.
- `--openai-model` (required): Model name (e.g., `gpt-4`).
- `--openai-api-key` (required): API key for the endpoint.
- `--source`: Filter by source/domain (optional).
- `--match-count`: Number of chunks to retrieve (default: 5).

#### 2. Crawl Website

Recursively crawl a website and store its content in the database.

```sh
python src/rag_cli.py crawl-website \
  --url "https://example.com" \
  [--max-depth 3] \
  [--chunk-size 5000]
```

- `--url` (required): The website URL to crawl.
- `--max-depth`: Recursion depth for internal links (default: 3).
- `--chunk-size`: Max chunk size for content splitting (default: 5000).

#### 3. Crawl GitHub Repo

Crawl a GitHub repository and store its content in the database.

```sh
python src/rag_cli.py crawl-github \
  --repo-url "https://github.com/user/repo" \
  [--branch main] \
  [--max-depth 3] \
  [--chunk-size 5000] \
  [--github-token <token>]
```

- `--repo-url` (required): The GitHub repository URL.
- `--branch`: Branch to crawl (default: `main`).
- `--max-depth`: Directory recursion depth (default: 3).
- `--chunk-size`: Max chunk size for content splitting (default: 5000).
- `--github-token`: GitHub token for private repos or higher rate limits (optional).

---

## **Help**

For detailed help on any subcommand, run:

```sh
python src/rag_cli.py --help
python src/rag_cli.py rag-query --help
python src/rag_cli.py crawl-website --help
python src/rag_cli.py crawl-github --help
```

---

## **Testing**

Run all tests with:

```sh
PYTHONPATH=. pytest tests/
```

---

## **Environment Variables**

- `DB_SECRET_NAME`, `AWS_REGION` (for AWS RDS/pgvector connection)
- `OPENAI_API_KEY` (for embedding, if not provided via CLI)
- `GITHUB_TOKEN` (optional, for private repo crawling)

---

## **Notes**

- All chunking, embedding, and DB logic is shared and maintained in `src/utils.py`.
- The CLI is fully tested and modular—see `tests/test_rag_cli.py` for examples. 