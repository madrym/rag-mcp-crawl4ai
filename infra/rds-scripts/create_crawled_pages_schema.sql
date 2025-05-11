-- Enable the pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the documentation chunks table (idempotent)
CREATE TABLE IF NOT EXISTS crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    UNIQUE(url, chunk_number)
);

-- Create an index for better vector similarity search performance (idempotent)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE tablename = 'crawled_pages' AND indexname = 'idx_crawled_pages_embedding_ivfflat'
    ) THEN
        CREATE INDEX idx_crawled_pages_embedding_ivfflat ON crawled_pages USING ivfflat (embedding vector_cosine_ops);
    END IF;
END$$;

-- Create an index on metadata for faster filtering (idempotent)
CREATE INDEX IF NOT EXISTS idx_crawled_pages_metadata ON crawled_pages USING gin (metadata);

-- Create an index on source for faster filtering (idempotent)
CREATE INDEX IF NOT EXISTS idx_crawled_pages_source ON crawled_pages ((metadata->>'source'));

-- Create or replace the function for vector search (idempotent)
CREATE OR REPLACE FUNCTION match_crawled_pages (
  p_query_embedding vector(1536),
  p_match_count int DEFAULT 10,
  p_filter jsonb DEFAULT '{}'::jsonb
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
  WHERE cp.metadata @> p_filter
  ORDER BY cp.embedding <=> p_query_embedding
  LIMIT p_match_count;
END;
$$;