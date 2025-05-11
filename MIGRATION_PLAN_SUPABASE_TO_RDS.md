# Migration Plan: MCP Server from Supabase to AWS RDS with pgvector

This document outlines the steps to migrate your Python-based MCP (Model Context Protocol) server from a Supabase backend to Amazon RDS for PostgreSQL with the pgvector extension. The goal is to maintain current functionality while leveraging AWS infrastructure.

## Table of Contents
1.  [Prerequisites](#prerequisites)
2.  [Phase 1: AWS Infrastructure Setup](#phase-1-aws-infrastructure-setup)
3.  [Phase 2: Database Schema Migration](#phase-2-database-schema-migration)
4.  [Phase 3: Application Code Adaptation](#phase-3-application-code-adaptation)
5.  [Phase 4: Configuration Update](#phase-4-configuration-update)
6.  [Phase 5: Testing and Validation](#phase-5-testing-and-validation)
7.  [Phase 6: Deployment](#phase-6-deployment)
8.  [Phase 7: Post-Migration](#phase-7-post-migration)
9.  [Security Best Practices Summary](#security-best-practices-summary)
10. [Performance Optimizations Summary](#performance-optimizations-summary)
11. [Transition and Implementation Strategy](#transition-and-implementation-strategy)

## Prerequisites

*   **AWS Account:** Access to an AWS account with permissions to create RDS instances, Security Groups, and Secrets Manager secrets.
*   **AWS CLI:** Installed and configured (optional, but helpful for scripting).
*   **PostgreSQL Client:** `psql` or a GUI tool like pgAdmin to interact with the database.
*   **Python Environment:** Your existing MCP server development environment.
*   **Existing Codebase:** The `mcp-crawl4ai-rag` repository.
*   **Familiarity:** Basic understanding of AWS VPC, IAM, RDS, and PostgreSQL.

---

## Phase 1: AWS Infrastructure Setup

### Step 1.1: Create RDS for PostgreSQL Instance

1.  **Navigate to RDS Console:** Log in to your AWS Management Console and go to the RDS service.
2.  **Create Database:**
    *   Click "Create database".
    *   Choose "Standard Create".
    *   Engine type: "PostgreSQL".
    *   Select a PostgreSQL version compatible with `pgvector` (e.g., PostgreSQL 15 or 16).
    *   **Templates:** Choose "Development/Test" for initial setup, or "Production" for production.
    *   **Settings:**
        *   `DB instance identifier`: A unique name (e.g., `mcp-crawl4ai-rds`).
        *   `Master username`: e.g., `mcp_admin`.
        *   `Master password`: Set a strong password (you will store this in Secrets Manager).
    *   **Instance configuration:** Choose an appropriate instance class (e.g., `db.t3.medium` for dev, `db.m5.large` or `db.r5.large` for prod).
    *   **Storage:** Configure storage type (General Purpose SSD - gp2 or gp3 recommended), allocated storage, and enable storage autoscaling.
    *   **Connectivity:**
        *   `Virtual Private Cloud (VPC)`: Select the VPC where your MCP server will run or a VPC that can be peered.
        *   `Subnet group`: Choose or create an appropriate DB subnet group (ensure subnets are private).
        *   `Public access`: Set to **No**. Access will be via the VPC.
        *   `VPC security group`: Choose "Create new" (e.g., `mcp-rds-sg`) or select an existing one. You'll configure this next.
        *   `Database port`: Default is `5432`.
    *   **Database authentication:** Choose "Password authentication" or "Password and IAM database authentication".
    *   **Additional configuration (expand):**
        *   `Initial database name`: e.g., `crawl4ai_db`.
        *   `DB parameter group`: Default is usually fine, but for `pgvector` ensure `shared_preload_libraries` can include `vector` if needed (often handled automatically when creating extension).
        *   Enable `Encryption at rest` using AWS KMS.
        *   Configure backups and maintenance windows.
3.  **Create Database:** Click the "Create database" button. Wait for the instance status to become "Available". Note the **Endpoint** and **Port** from the "Connectivity & security" tab.

**Assistance Notes:**
*   Start with a smaller instance for development to save costs.
*   Ensure your RDS instance is in private subnets for security.

### Step 1.2: Configure Security Groups

1.  **Navigate to VPC Console -> Security Groups.**
2.  Find the security group associated with your RDS instance (e.g., `mcp-rds-sg`).
3.  **Edit Inbound Rules:**
    *   Click "Edit inbound rules".
    *   "Add rule":
        *   `Type`: `PostgreSQL`.
        *   `Protocol`: `TCP`.
        *   `Port range`: `5432`.
        *   `Source`:
            *   If your MCP server runs on EC2/ECS/Fargate: Select the security group of your application.
            *   For local development: Your IP address (select "My IP" or specify a range). **Restrict this as much as possible.**
4.  **Save rules.**

**Assistance Notes:**
*   Be very specific with source IPs/SGs to minimize exposure.

### Step 1.3: Enable `pgvector` Extension

1.  **Connect to RDS using `psql`:**
    ```bash
    psql --host=<your-rds-endpoint> --port=5432 --username=<your-master-username> --dbname=<your-initial-db-name>
    ```
    Enter the master password when prompted.

2.  **Enable the extension:**
    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    ```

3.  **Verify installation:**
    ```sql
    \dx vector
    ```
    You should see `vector` listed.

**Assistance Notes:**
*   The RDS endpoint can be found on the RDS instance's "Connectivity & security" tab.
*   If `CREATE EXTENSION` fails due to permissions, ensure you're connected as the master user or a user with sufficient privileges.

### Step 1.4: Set up AWS Secrets Manager

1.  **Navigate to Secrets Manager Console.**
2.  **Store a new secret:**
    *   `Secret type`: "Credentials for RDS database".
    *   `User name`: The master username for your RDS instance (e.g., `mcp_admin`).
    *   `Password`: The master password.
    *   `Encryption key`: Default `aws/secretsmanager` is fine.
    *   `Database`: Select your newly created RDS instance.
    *   Click "Next".
3.  **Secret name:** e.g., `mcp/crawl4ai/db_credentials`. Add a description.
4.  **Configure rotation (Optional):** You can set up automatic rotation if desired.
5.  **Store secret.** Note the **Secret ARN** or **Secret name**.

**Assistance Notes:**
*   The IAM role/user running your MCP server application will need `secretsmanager:GetSecretValue` permission for this secret.

---

## Phase 2: Database Schema Migration

### Step 2.1: Connect to RDS Instance (if not already connected)

Use `psql` as described in Step 1.3.

### Step 2.2: Execute Schema SQL

1.  Take the contents of your existing `crawled_pages.sql` file. This file contains the `CREATE TABLE crawled_pages` statement, index creations (including the `pgvector` index), and the `match_crawled_pages` function.

    ```sql
    -- Ensure this is your target database
    -- \c crawl4ai_db 

    -- Enable the pgvector extension (if not already done, but Step 1.3 covers this)
    -- CREATE EXTENSION IF NOT EXISTS vector; 

    -- Create the documentation chunks table
    CREATE TABLE IF NOT EXISTS crawled_pages (
        id bigserial primary key,
        url varchar not null,
        chunk_number integer not null,
        content text not null,
        metadata jsonb not null default \'{}\'::jsonb,
        embedding vector(1536), -- OpenAI embeddings are 1536 dimensions
        created_at timestamp with time zone default timezone(\'utc\'::text, now()) not null,
        UNIQUE(url, chunk_number)
    );

    -- Create an index for better vector similarity search performance
    -- Choose one: IVFFlat or HNSW (HNSW often preferred for new projects)
    -- CREATE INDEX IF NOT EXISTS idx_crawled_pages_embedding_ivfflat ON crawled_pages USING ivfflat (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS idx_crawled_pages_embedding_hnsw ON crawled_pages USING hnsw (embedding vector_cosine_ops);


    -- Create an index on metadata for faster filtering
    CREATE INDEX IF NOT EXISTS idx_crawled_pages_metadata ON crawled_pages USING gin (metadata);
    CREATE INDEX IF NOT EXISTS idx_crawled_pages_source ON crawled_pages ((metadata->>\'source\'));

    -- Create a function to search for documentation chunks
    CREATE OR REPLACE FUNCTION match_crawled_pages (
      p_query_embedding vector(1536),
      p_match_count int DEFAULT 10,
      p_filter jsonb DEFAULT \'{}\'::jsonb
    ) RETURNS TABLE (
      id bigint,
      url varchar,
      chunk_number integer,
      content text,
      metadata jsonb,
      similarity float
    )
    LANGUAGE plpgsql
    AS $$
    BEGIN
      RETURN QUERY
      SELECT
        cp.id,
        cp.url,
        cp.chunk_number,
        cp.content,
        cp.metadata,
        1 - (cp.embedding <=> p_query_embedding) AS similarity
      FROM crawled_pages cp
      WHERE cp.metadata @> p_filter -- Ensure this filtering logic is what you need
      ORDER BY cp.embedding <=> p_query_embedding
      LIMIT p_match_count;
    END;
    $$;

    -- RLS policies (review if needed for RDS, often managed by DB users/roles directly)
    -- ALTER TABLE crawled_pages ENABLE ROW LEVEL SECURITY;
    -- CREATE POLICY "Allow public read access" ON crawled_pages FOR SELECT TO public USING (true);
    -- For RDS, typically you grant explicit user permissions rather than broad RLS for public.
    -- Example: GRANT SELECT, INSERT, UPDATE, DELETE ON crawled_pages TO your_app_user;
    --          GRANT EXECUTE ON FUNCTION match_crawled_pages TO your_app_user;
    ```

2.  Execute this SQL in your `psql` session connected to the RDS instance.

**Assistance Notes:**
*   The RLS (Row Level Security) policies from Supabase might not be directly applicable or desired in RDS. Instead, rely on standard PostgreSQL user roles and `GRANT` permissions. Create a dedicated application user with least privilege.
*   Consider using `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` for idempotency.
*   I've included HNSW as an alternative for the vector index, as it's often a good default. Test which index type (`ivfflat` or `hnsw`) works best for your data and query patterns.

---

## Phase 3: Application Code Adaptation

### Step 3.1: Update Dependencies

Add `psycopg2-binary` (for PostgreSQL connection) and `boto3` (for AWS SDK, including Secrets Manager) to your `pyproject.toml`:

```toml
# pyproject.toml
[project]
# ... existing config ...
dependencies = [
    "crawl4ai==0.6.2",
    "mcp==1.7.1",
    # "supabase==2.15.1", # Remove or comment out
    "openai==1.71.0",
    "dotenv==0.9.9",
    "psycopg2-binary",  # Add this
    "boto3"             # Add this
]
```
Then, update your environment:
```bash
uv pip install -e .
```

### Step 3.2: Modify `src/utils.py`

Replace Supabase client interactions with `psycopg2` and `boto3`.

**1. Database Connection (`get_db_connection` replacing `get_supabase_client`):**
```python
# src/utils.py
import os
import json
import psycopg2
import boto3 # For AWS Secrets Manager
from psycopg2.extras import Json, DictCursor # For easy JSON handling and dict results
from typing import List, Dict, Any, Tuple, Optional # Ensure these are imported

# (Keep existing imports like openai, load_dotenv)
# from dotenv import load_dotenv
# load_dotenv() # Ensure .env is loaded

_db_conn: Optional[psycopg2.extensions.connection] = None # Type hint for connection

def get_secret(secret_name: str, region_name: str) -> Dict[str, str]:
    """Retrieve secret from AWS Secrets Manager."""
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return json.loads(secret)
        else:
            # Handle binary secret if necessary
            decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])
            return json.loads(decoded_binary_secret)
    except Exception as e:
        print(f"Error retrieving secret {secret_name}: {e}")
        raise

def get_db_connection() -> psycopg2.extensions.connection:
    """
    Get a PostgreSQL database connection.
    Uses a cached connection if available.
    Fetches credentials from environment variables or AWS Secrets Manager.
    """
    global _db_conn
    if _db_conn and not _db_conn.closed:
        return _db_conn

    db_user, db_password, db_host, db_port_str, db_name_val = (None,) * 5

    aws_secret_name = os.getenv("DB_SECRET_NAME")
    aws_region = os.getenv("AWS_REGION")

    if aws_secret_name and aws_region:
        print(f"Fetching DB credentials from AWS Secrets Manager: {aws_secret_name}")
        try:
            secret_data = get_secret(aws_secret_name, aws_region)
            db_user = secret_data.get('username')
            db_password = secret_data.get('password')
            db_host = secret_data.get('host') # RDS secret usually includes host
            db_port_str = str(secret_data.get('port', '5432'))
            db_name_val = secret_data.get('dbname')
        except Exception as e:
            print(f"Failed to get credentials from Secrets Manager: {e}. Falling back to env vars if configured.")
    
    # Fallback or direct env var usage
    if not all([db_user, db_password, db_host, db_port_str, db_name_val]):
        print("Attempting to use direct DB credentials from environment variables.")
        db_host = db_host or os.getenv("DB_HOST")
        db_port_str = db_port_str or os.getenv("DB_PORT", "5432")
        db_name_val = db_name_val or os.getenv("DB_NAME")
        db_user = db_user or os.getenv("DB_USER")
        db_password = db_password or os.getenv("DB_PASSWORD")

    if not all([db_host, db_port_str, db_name_val, db_user, db_password]):
        raise ValueError(
            "Database connection parameters (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, or DB_SECRET_NAME, AWS_REGION) are not fully configured."
        )

    try:
        print(f"Connecting to database: host={db_host}, port={db_port_str}, dbname={db_name_val}, user={db_user}")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port_str,
            dbname=db_name_val,
            user=db_user,
            password=db_password,
            # Consider adding sslmode='require' for enforcing SSL
        )
        _db_conn = conn
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        _db_conn = None
        raise

def close_db_connection():
    global _db_conn
    if _db_conn and not _db_conn.closed:
        _db_conn.close()
        _db_conn = None
        print("Database connection closed.")

# (Keep create_embeddings_batch, create_embedding, generate_contextual_embedding, process_chunk_with_context)
# ... (your existing OpenAI and embedding related functions)
```

**2. Document Insertion (`add_documents_batch`):**
```python
# src/utils.py
# ... (after get_db_connection and embedding functions)

def add_documents_batch(
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """
    Add documents to the RDS crawled_pages table in batches.
    Deletes existing records for the given URLs before inserting to prevent duplicates.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        unique_urls = list(set(urls))
        if unique_urls:
            print(f"Deleting existing records for {len(unique_urls)} URLs...")
            # psycopg2 adapts a list of strings to a postgres array for ANY
            delete_query = "DELETE FROM crawled_pages WHERE url = ANY(%s);"
            cursor.execute(delete_query, (unique_urls,))
            print(f"{cursor.rowcount} existing records deleted for specified URLs.")
            conn.commit()

        model_choice = os.getenv("MODEL_CHOICE")
        use_contextual_embeddings = bool(model_choice)
        
        total_inserted_count = 0
        for i in range(0, len(contents), batch_size):
            batch_end = min(i + batch_size, len(contents))
            print(f"Processing DB batch {i//batch_size + 1}: items {i+1} to {batch_end}")

            batch_urls_slice = urls[i:batch_end]
            batch_chunk_numbers_slice = chunk_numbers[i:batch_end]
            batch_contents_slice = contents[i:batch_end]
            batch_metadatas_slice = metadatas[i:batch_end]
            
            content_for_embedding = batch_contents_slice
            if use_contextual_embeddings:
                # This part generates text for embeddings (OpenAI calls via your functions)
                # Ensure generate_contextual_embedding is robust
                temp_contextual_contents = []
                for j_idx, chunk_content in enumerate(batch_contents_slice):
                    current_url = batch_urls_slice[j_idx]
                    full_doc = url_to_full_document.get(current_url, "")
                    contextual_text, _ = generate_contextual_embedding(full_doc, chunk_content)
                    temp_contextual_contents.append(contextual_text)
                content_for_embedding = temp_contextual_contents

            batch_embeddings = create_embeddings_batch(content_for_embedding)
            if not batch_embeddings or len(batch_embeddings) != len(content_for_embedding):
                print(f"Warning: Embedding count mismatch for batch starting at index {i}. Expected {len(content_for_embedding)}, got {len(batch_embeddings) if batch_embeddings else 0}.")
                # Decide how to handle: skip batch, or skip items without embeddings
                # For now, we'll try to insert only those with embeddings
                # This assumes create_embeddings_batch returns a list of same length, with None/empty for failures

            batch_data_to_insert = []
            for j in range(len(batch_contents_slice)):
                # Ensure we have an embedding for this item
                if j >= len(batch_embeddings) or not batch_embeddings[j]:
                    print(f"Skipping doc URL {batch_urls_slice[j]} chunk {batch_chunk_numbers_slice[j]} due to missing/failed embedding.")
                    continue
                
                current_metadata = batch_metadatas_slice[j] if batch_metadatas_slice and j < len(batch_metadatas_slice) else {}

                data_tuple = (
                    batch_urls_slice[j],
                    batch_chunk_numbers_slice[j],
                    batch_contents_slice[j],       # Store original content
                    Json(current_metadata),        # Use psycopg2.extras.Json
                    batch_embeddings[j]            # Embedding vector (list of floats)
                )
                batch_data_to_insert.append(data_tuple)
            
            if batch_data_to_insert:
                insert_query = """
                    INSERT INTO crawled_pages (url, chunk_number, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (url, chunk_number) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        created_at = timezone(\'utc\'::text, now());
                """
                # Using ON CONFLICT for robustness, though explicit delete was done earlier
                # For batch insert, psycopg2.extras.execute_values is efficient
                from psycopg2.extras import execute_values
                execute_values(cursor, insert_query, batch_data_to_insert)
                conn.commit()
                total_inserted_count += len(batch_data_to_insert)
                print(f"DB Batch {i//batch_size + 1}: Inserted/Updated {len(batch_data_to_insert)} records.")
        
        print(f"Successfully inserted/updated {total_inserted_count} total records.")

    except psycopg2.Error as e:
        print(f"Database error in add_documents_batch: {e}")
        if conn: conn.rollback()
        raise 
    except Exception as e:
        print(f"General error in add_documents_batch: {e}")
        if conn: conn.rollback()
        raise
    # finally:
        # Connection is managed globally for now by _db_conn
        # if cursor: cursor.close()
```

**3. Vector Similarity Search (`search_documents`):**
```python
# src/utils.py
# ... (after add_documents_batch)

def search_documents(
    query: str,
    match_count: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None # Made optional
) -> List[Dict[str, Any]]:
    """
    Search for documents in RDS using vector similarity via the match_crawled_pages function.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor) # Results as dictionaries

        query_embedding = create_embedding(query)
        if not query_embedding:
            print("Failed to generate query embedding for search.")
            return []

        params_for_sql = {
            'p_query_embedding': query_embedding,
            'p_match_count': match_count,
            'p_filter': Json(filter_metadata if filter_metadata else {}) # Pass Json object
        }
        
        sql_function_call = "SELECT * FROM match_crawled_pages(%(p_query_embedding)s, %(p_match_count)s, %(p_filter)s);"
        
        cursor.execute(sql_function_call, params_for_sql)
        results = cursor.fetchall() # List of DictRow objects

        return [dict(row) for row in results] # Convert DictRow to plain dict

    except psycopg2.Error as e:
        print(f"Database error in search_documents: {e}")
        if conn: conn.rollback() # Good practice, though SELECTs don't usually need it
        return []
    except Exception as e:
        print(f"General error in search_documents: {e}")
        return []
    # finally:
        # Connection is managed globally
        # if cursor: cursor.close()
```

**Assistance Notes for `src/utils.py`:**
*   The provided `get_db_connection` uses a single global cached connection. For production, **implement proper connection pooling** (e.g., using `psycopg2.pool.SimpleConnectionPool` or `ThreadedConnectionPool`). The pool should be initialized once (e.g., in the lifespan manager of your MCP server) and connections acquired/released from it.
*   Error handling is basic; expand as needed.
*   The `add_documents_batch` uses `ON CONFLICT` which is generally safer than delete then insert if multiple processes might interact, though the explicit delete for given URLs is still there.
*   `psycopg2` is synchronous. If your MCP server is heavily async (`asyncio`), consider using an async library like `asyncpg` or running sync DB calls in a thread pool executor to avoid blocking the event loop.
    ```python
    # Example for async:
    # import asyncio
    # loop = asyncio.get_event_loop()
    # conn = await loop.run_in_executor(None, get_db_connection) 
    # # Then use conn with other run_in_executor calls for cursor operations
    ```

### Step 3.3: Modify `src/crawl4ai_mcp.py`

**Adapt `get_available_sources` tool:**
This tool in `src/crawl4ai_mcp.py` needs to use the new DB connection.

```python
# src/crawl4ai_mcp.py
# ... (other imports)
# Make sure these are available:
import json
from mcp import FastMCP, Context # Assuming Context is from mcp
from src.utils import get_db_connection # Your new function
# from psycopg2.extras import DictCursor # If you want dict results directly

# ... (FastMCP initialization, other tools)

@mcp.tool()
async def get_available_sources(ctx: Context) -> str: # Added ctx type hint
    """
    Get all available sources based on unique source metadata values from RDS.
    """
    # For async tools with sync DB calls, use run_in_executor
    import asyncio
    loop = asyncio.get_event_loop()

    def _get_sources_sync():
        conn = None
        try:
            conn = get_db_connection()
            # cursor = conn.cursor(cursor_factory=DictCursor) # For list of dicts
            cursor = conn.cursor() # For list of tuples

            query = """
                SELECT DISTINCT metadata->>\'source\' AS source 
                FROM crawled_pages 
                WHERE metadata->>\'source\' IS NOT NULL AND metadata->>\'source\' <> \'\';
            """
            cursor.execute(query)
            results = cursor.fetchall() # List of tuples, e.g., [(\'example.com\',), (\'another.dev\',)]
            
            sources = sorted([row[0] for row in results if row[0] is not None])
            return sources
        finally:
            if cursor: cursor.close()
            # Do not close global connection here; manage it via close_db_connection() at app shutdown
            # or if using a pool, release connection back to pool.

    try:
        sources = await loop.run_in_executor(None, _get_sources_sync)
        return json.dumps({"sources": sources})
    except Exception as e:
        print(f"Error in get_available_sources: {e}")
        # Ensure to log the exception e
        return json.dumps({"error": str(e), "sources": []})

# ... (other tools like crawl_single_page, smart_crawl_url, perform_rag_query)
# These tools primarily call functions in utils.py (like add_documents_batch, search_documents),
# so they might not need direct changes if those util functions are correctly refactored.
# However, they are async, so ensure any direct DB calls within them also use run_in_executor.

# Ensure the lifespan manager in crawl4ai_mcp.py calls close_db_connection on shutdown
# @asynccontextmanager
# async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
#     # ... setup ...
#     try:
#         yield your_context_object # Pass context
#     finally:
#         # ... other cleanup ...
#         print("Closing database connection during server shutdown.")
#         close_db_connection() # Call the function from utils.py
```

**Assistance Notes for `src/crawl4ai_mcp.py`:**
*   The `get_available_sources` example shows how to use `loop.run_in_executor` for running synchronous `psycopg2` code in an async MCP tool. Apply this pattern if other tools make direct synchronous DB calls.
*   Make sure the `lifespan` context manager in `crawl4ai_mcp.py` calls `close_db_connection()` from `utils.py` during server shutdown to properly close the cached global connection. If you implement connection pooling, the lifespan manager would be responsible for initializing and closing the pool.

---

## Phase 4: Configuration Update

### Step 4.1: Update `.env` file

Modify your `.env` file (and update `.env.example` accordingly):

```dotenv
# MCP Server Configuration (if used)
# HOST=0.0.0.0
# PORT=8051
# TRANSPORT=sse

# OpenAI API Configuration (remains the same)
OPENAI_API_KEY=your_openai_api_key
MODEL_CHOICE= # e.g., gpt-4.1-nano, or leave blank for no contextual embeddings

# --- AWS Configuration ---
AWS_REGION=your-aws-region # e.g., us-east-1

# Option 1: For AWS Secrets Manager (Recommended)
DB_SECRET_NAME=mcp/crawl4ai/db_credentials # Name of the secret in AWS Secrets Manager

# Option 2: For Direct DB Credentials (Less secure, use for local dev if Secrets Manager is not set up)
# Ensure DB_SECRET_NAME is commented out or empty if using these:
# DB_HOST=your-rds-instance-endpoint.region.rds.amazonaws.com
# DB_PORT=5432
# DB_NAME=crawl4ai_db
# DB_USER=your_app_db_user # Best to use a dedicated app user, not master
# DB_PASSWORD=your_app_db_password

# --- Remove or comment out Supabase variables ---
# SUPABASE_URL=
# SUPABASE_SERVICE_KEY=
```

**Assistance Notes:**
*   If your MCP server runs in an AWS environment (EC2, ECS, Fargate, Lambda), ensure the IAM role attached to the compute resource has `secretsmanager:GetSecretValue` permission for the `DB_SECRET_NAME`.
*   For local development, you might temporarily use direct `DB_HOST`, `DB_USER`, etc., variables, but aim to use `DB_SECRET_NAME` where possible.

---

## Phase 5: Testing and Validation

### Step 5.1: Local Testing

1.  Ensure your local environment has access to the RDS instance (Security Group inbound rule for your IP).
2.  Set the necessary environment variables in your local `.env` file.
3.  Run your MCP server locally: `uv run src/crawl4ai_mcp.py`.
4.  Thoroughly test each MCP tool:
    *   `crawl_single_page`: Crawl a test page.
    *   `smart_crawl_url`: Crawl a test website/sitemap/txt file.
    *   `get_available_sources`: Check if it returns expected sources after crawling.
    *   `perform_rag_query`: Perform semantic searches on the crawled content.

### Step 5.2: Data Verification

*   After crawling, connect to your RDS instance using `psql` or pgAdmin.
*   Inspect the `crawled_pages` table:
    ```sql
    SELECT url, chunk_number, left(content, 100) as content_preview, metadata FROM crawled_pages LIMIT 10;
    SELECT count(*) FROM crawled_pages;
    SELECT metadata->>\'source\', count(*) FROM crawled_pages GROUP BY metadata->>\'source\';
    ```
*   Verify that embeddings are being stored (e.g., check if the `embedding` column is not NULL).
*   Ensure RAG queries return relevant results.

### Step 5.3: Performance Testing (Optional)

*   For `perform_rag_query`, check the response time. If slow, review `pgvector` index strategies (Phase 10).
*   Monitor RDS CPU and memory usage during crawling and querying via CloudWatch.

---

## Phase 6: Deployment

### Step 6.1: Update Deployment Scripts/Configuration

*   **Docker:** If you use Docker, update your `Dockerfile` if needed (though changes are mainly in Python code and env vars). Ensure your Docker run command or ECS task definition passes the new environment variables (especially `AWS_REGION`, `DB_SECRET_NAME`, or the direct DB vars).
*   **EC2/ECS/Fargate:**
    *   Ensure the IAM role for your compute instances/tasks has permissions for Secrets Manager (`secretsmanager:GetSecretValue` on the specific secret) and any other AWS services it needs (e.g., CloudWatch Logs).
    *   Update environment variables in your task definitions (ECS) or instance user data/launch templates (EC2).

### Step 6.2: Deploy Updated Application

Follow your standard deployment process to roll out the updated MCP server.

---

## Phase 7: Post-Migration

### Step 7.1: Monitor System

*   Closely monitor application logs for any database-related errors.
*   Monitor RDS instance metrics in CloudWatch (CPU, memory, disk space, connections, query latency).
*   Monitor AWS Secrets Manager logs if issues arise with credential fetching.

### Step 7.2: Decommission Supabase (After Thorough Validation)

Once you are confident that the new RDS-backed system is stable and performing correctly, and all necessary data has been migrated (if applicable), you can plan to decommission your Supabase database to avoid ongoing costs.
1.  Backup any final data from Supabase if needed.
2.  Remove Supabase credentials from your application's active configuration.
3.  Delete the Supabase project or database.

---

## Security Best Practices Summary

*   **IAM Database Authentication:** Consider for enhanced security over passwords.
*   **AWS Secrets Manager:** Use for storing DB credentials.
*   **VPC & Private Subnets:** Run RDS in private subnets.
*   **Security Groups:** Restrict access tightly.
*   **Encryption:** Enable encryption at rest (RDS) and enforce SSL/TLS for connections in transit.
*   **Least Privilege:** Use dedicated database users with minimal necessary permissions. Grant specific IAM permissions to your application role.
*   **Patching & Updates:** Keep RDS and application dependencies updated.
*   **Logging & Monitoring:** Use CloudWatch for RDS and application logs.

---

## Performance Optimizations Summary

*   **RDS Instance Sizing:** Choose appropriate instance class.
*   **`pgvector` Index Tuning:** Experiment with `ivfflat` vs. `hnsw` and their parameters. Regularly `VACUUM ANALYZE`.
*   **Connection Pooling:** **Implement this for production.** Use `psycopg2.pool` or RDS Proxy.
*   **Batch Operations:** Continue using for inserts.
*   **Asynchronous Database Operations:** Use `asyncpg` or `run_in_executor` for `psycopg2` in async code.
*   **Read Replicas:** Consider for read-heavy workloads.

---

## Transition and Implementation Strategy

This plan outlines how to make the switch from Supabase to AWS RDS.

### A. Development & Initial Testing (Isolated Environment)

1.  **Setup Dev RDS:** Provision a new, separate RDS instance specifically for development and testing (as per Phase 1). This avoids any impact on your current Supabase setup.
2.  **Code Implementation:** Implement all the code changes (Phase 3) in a dedicated branch of your repository.
3.  **Configuration:** Use a local `.env` file pointing to this dev RDS instance.
4.  **Thorough Testing:** Execute all test cases (Phase 5). Iterate on code changes until all functionalities work correctly against the dev RDS.

### B. Staging Environment (Recommended)

1.  **Setup Staging RDS:** If you have a staging environment, provision an RDS instance for it, mirroring production configuration as closely as possible.
2.  **Deploy:** Deploy the updated application branch to your staging environment.
3.  **Data Seeding/Migration (Test):** If you have existing data in Supabase that needs to be present for staging tests, perform a test migration of a subset of this data to the staging RDS. (See "Data Migration Strategy" below).
4.  **Comprehensive Testing:** Conduct user acceptance testing (UAT), performance tests, and integration tests in staging.

### C. Production Cutover

Schedule a maintenance window to minimize impact.

**Data Migration Strategy for Existing Data (If any in Supabase):**
If you have critical data in your Supabase `crawled_pages` table that needs to be moved to the new RDS production instance, you'll need a one-time migration script.

*   **Script Outline:**
    1.  Connect to your Supabase database (using `supabase-py` or `psycopg2` if Supabase allows direct PG connections with existing credentials).
    2.  Fetch data from the `crawled_pages` table in batches.
    3.  For each batch, transform the data if necessary to match the format expected by your new `add_documents_batch` function (it should be very similar).
    4.  Use your *new* `add_documents_batch` function (which writes to RDS) to insert this data into the production RDS instance.
*   **Testing:** Test this migration script thoroughly against your dev/staging RDS instances first.
*   **Timing:** Perform this data migration during the scheduled maintenance window *before* switching traffic to the new application.

**Production Cutover Options:**

*   **Option 1: Big Bang (Common for such migrations):**
    1.  **Announce Maintenance:** Inform users of the scheduled downtime.
    2.  **Stop Old System:** Prevent any new data from being written to Supabase (e.g., stop the old application or put it in read-only mode if possible).
    3.  **(If applicable) Final Data Migration:** Run your data migration script to move all existing data from Supabase to the new production RDS instance.
    4.  **Deploy New Application:** Deploy the updated application version configured to use the new production RDS instance.
    5.  **Sanity Checks:** Perform critical path testing to ensure the system is working correctly with RDS.
    6.  **Lift Maintenance:** Announce the system is back online.
    7.  **Monitor:** Closely monitor the system (Phase 7.1).

*   **Option 2: Phased Rollout / Read-Only Transition (More Complex):**
    *   This is harder if writes are involved. One approach could be:
        1.  Migrate existing data to RDS.
        2.  Deploy the new application to read from RDS but still write to Supabase (requires temporary dual-write logic or decision logic, adding complexity).
        3.  Once reads are stable, switch writes to RDS.
    *   This often isn't worth the complexity for a backend database switch unless zero downtime is an absolute, hard requirement and the application can support it.

### D. Post-Cutover Monitoring & Rollback Plan

1.  **Intensive Monitoring:** Immediately after cutover, closely monitor application logs, RDS performance metrics, and user-reported issues.
2.  **Rollback Plan:** Have a clear, tested rollback plan. This typically involves:
    *   Reverting the application deployment to the previous version (that uses Supabase).
    *   Ensuring Supabase still has the data integrity up to the point of cutover (if dual writes weren't implemented, data written to RDS during a failed cutover attempt would be lost or need re-migration).
    *   Re-pointing DNS or load balancers if applicable.
    *   The complexity of rollback depends on how much new data might have been written to RDS before a rollback decision is made.

---

This detailed plan should provide a clear path for your migration. Remember to adapt it to your specific environment and requirements. Good luck! 