"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
import psycopg2
import boto3
from psycopg2.extras import Json, DictCursor
import openai
import re

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")

_db_conn: Optional[psycopg2.extensions.connection] = None

def get_secret(secret_name: str, region_name: str) -> Dict[str, str]:
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return json.loads(secret)
        else:
            import base64
            decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])
            return json.loads(decoded_binary_secret)
    except Exception as e:
        print(f"Error retrieving secret {secret_name}: {e}")
        raise

def get_db_connection() -> psycopg2.extensions.connection:
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
            db_host = secret_data.get('host')
            db_port_str = str(secret_data.get('port', '5432'))
            db_name_val = secret_data.get('dbname')
        except Exception as e:
            print(f"Failed to get credentials from Secrets Manager: {e}. Falling back to env vars if configured.")

    if not all([db_user, db_password, db_host, db_port_str, db_name_val]):
        missing = []
        if not db_host: missing.append("DB_HOST")
        if not db_port_str: missing.append("DB_PORT")
        if not db_name_val: missing.append("DB_NAME")
        if not db_user: missing.append("DB_USER")
        if not db_password: missing.append("DB_PASSWORD")
        if missing:
            raise ValueError(
                f"Database connection parameters missing: {', '.join(missing)}. "
                "Please set them in your environment variables or AWS Secrets Manager."
            )

    try:
        print(f"Connecting to database: host={db_host}, port={db_port_str}, dbname={db_name_val}, user={db_user}")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port_str,
            dbname=db_name_val,
            user=db_user,
            password=db_password,
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

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
        
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small", # Hardcoding embedding model for now, will change this later to be more dynamic
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error creating batch embeddings: {e}")
        # Return empty embeddings if there's an error
        return [[0.0] * 1536 for _ in range(len(texts))]

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

def safe_execute_values(cursor, query, data):
    """
    Helper for psycopg2.extras.execute_values that enforces VALUES %s pattern.
    Raises ValueError if the query does not use VALUES %s.
    """
    if "VALUES %s" not in query:
        raise ValueError("For execute_values, query must use VALUES %s")
    from psycopg2.extras import execute_values
    return execute_values(cursor, query, data)

def add_documents_batch(
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        unique_urls = list(set(urls))
        if unique_urls:
            print(f"Deleting existing records for {len(unique_urls)} URLs...")
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
            batch_data_to_insert = []
            for j in range(len(batch_contents_slice)):
                if j >= len(batch_embeddings) or not batch_embeddings[j]:
                    print(f"Skipping doc URL {batch_urls_slice[j]} chunk {batch_chunk_numbers_slice[j]} due to missing/failed embedding.")
                    continue
                current_metadata = batch_metadatas_slice[j] if batch_metadatas_slice and j < len(batch_metadatas_slice) else {}
                data_tuple = (
                    batch_urls_slice[j],
                    batch_chunk_numbers_slice[j],
                    batch_contents_slice[j],
                    Json(current_metadata),
                    batch_embeddings[j]
                )
                batch_data_to_insert.append(data_tuple)
            if batch_data_to_insert:
                # IMPORTANT: For execute_values, use VALUES %s (not VALUES (%s, %s, ...))
                insert_query = """
                    INSERT INTO crawled_pages (url, chunk_number, content, metadata, embedding)
                    VALUES %s
                    ON CONFLICT (url, chunk_number) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        created_at = timezone('utc'::text, now());
                """
                safe_execute_values(cursor, insert_query, batch_data_to_insert)
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

def search_documents(
    query: str,
    match_count: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        query_embedding = create_embedding(query)
        if not query_embedding:
            print("Failed to generate query embedding for search.")
            return []
        params_for_sql = {
            'p_query_embedding': query_embedding,
            'p_match_count': match_count,
            'p_filter': Json(filter_metadata if filter_metadata else {})
        }
        sql_function_call = "SELECT * FROM match_crawled_pages(%(p_query_embedding)s::vector, %(p_match_count)s, %(p_filter)s);"
        cursor.execute(sql_function_call, params_for_sql)
        results = cursor.fetchall()
        return [dict(row) for row in results]
    except psycopg2.Error as e:
        print(f"Database error in search_documents: {e}")
        if conn: conn.rollback()
        return []
    except Exception as e:
        print(f"General error in search_documents: {e}")
        return []

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> list:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> dict:
    """
    Extracts headers and stats from a chunk.
    Args:
        chunk: Markdown chunk
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''
    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }