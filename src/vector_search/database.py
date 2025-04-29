"""Database providers for vector search."""
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from supabase import Client, create_client

from .config import Config


class BaseDatabase(ABC):
    """Base class for database providers."""

    def __init__(self, config: Config):
        """Initialize database provider.

        Args:
            config: Configuration instance
        """
        self.config = config

    @abstractmethod
    def initialize(self) -> None:
        """Initialize database schema and required tables."""
        pass

    @abstractmethod
    def store_embeddings(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        """Store text chunks and their embeddings.

        Args:
            chunks: List of chunk dictionaries containing text and metadata
            embeddings: Numpy array of embeddings
        """
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, limit: int = 5) -> List[Dict]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return

        Returns:
            List of chunk dictionaries sorted by similarity
        """
        pass


class PostgresDatabase(BaseDatabase):
    """PostgreSQL database provider with pgvector extension."""

    def __init__(self, config: Config):
        """Initialize PostgreSQL database connection.

        Args:
            config: Configuration instance
        """
        super().__init__(config)
        self.vector_dim = config.vector_dim
        self.conn = psycopg2.connect(
            dbname=config.db_name,
            user=config.db_user,
            password=config.db_password,
            host=config.db_host or "localhost",
            port=config.db_port or 5432
        )

    def initialize(self) -> None:
        """Initialize database schema and enable pgvector extension."""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create chunks table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({self.vector_dim}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create vector similarity search index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
                ON chunks 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            self.conn.commit()

    def store_embeddings(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        """Store text chunks and their embeddings in PostgreSQL.

        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
        """
        with self.conn.cursor() as cur:
            values = [
                (chunk["text"], chunk["metadata"], embedding.tolist())
                for chunk, embedding in zip(chunks, embeddings)
            ]
            
            execute_values(
                cur,
                """
                INSERT INTO chunks (text, metadata, embedding)
                VALUES %s
                """,
                values,
                template="(%s, %s::jsonb, %s::vector)"
            )
            
            self.conn.commit()

    def search(self, query_embedding: np.ndarray, limit: int = 5) -> List[Dict]:
        """Search for similar chunks using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return

        Returns:
            List of chunk dictionaries sorted by similarity
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT text, metadata, 1 - (embedding <=> %s) as similarity
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (query_embedding.tolist(), query_embedding.tolist(), limit)
            )
            
            results = []
            for text, metadata, similarity in cur.fetchall():
                results.append({
                    "text": text,
                    "metadata": metadata,
                    "similarity": float(similarity)
                })
                
            return results


class SupabaseDatabase(BaseDatabase):
    """Supabase database provider."""

    def __init__(self, config: Config):
        """Initialize Supabase client.

        Args:
            config: Configuration instance
        """
        super().__init__(config)
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key must be provided")
            
        self.client = create_client(self.url, self.key)
        self.table_name = "chunks"

    def initialize(self) -> None:
        """Initialize database schema using Supabase SQL editor.

        Note: This method assumes you have already created the necessary
        tables and enabled the pgvector extension in your Supabase project.
        """
        pass  # Schema should be initialized through Supabase dashboard

    def store_embeddings(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        """Store text chunks and their embeddings in Supabase.

        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
        """
        data = [
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": embedding.tolist()
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        
        self.client.table(self.table_name).insert(data).execute()

    def search(self, query_embedding: np.ndarray, limit: int = 5) -> List[Dict]:
        """Search for similar chunks using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return

        Returns:
            List of chunk dictionaries sorted by similarity
        """
        response = (
            self.client.rpc(
                "match_chunks",
                {
                    "query_embedding": query_embedding.tolist(),
                    "match_threshold": 0.0,
                    "match_count": limit
                }
            )
            .execute()
        )
        
        results = []
        for item in response.data:
            results.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "similarity": item["similarity"]
            })
            
        return results 