"""Database providers for vector search."""
import os
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from supabase import Client, create_client


class BaseDatabase(ABC):
    """Base class for database providers."""

    TABLE_SCHEMA = """
        id SERIAL PRIMARY KEY,
        embedding vector(%d),
        text TEXT NOT NULL,
        metadata JSONB,
        date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    """

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

    def __init__(
        self,
        dbname: str = None,
        user: str = None,
        password: str = None,
        host: str = "localhost",
        port: int = 5432,
        vector_dim: int = 1536,
        table_name: str = "chunks"
    ):
        """Initialize PostgreSQL database connection.

        Args:
            dbname: Database name
            user: Database user
            password: Database password
            host: Database host
            port: Database port
            vector_dim: Dimension of vectors to store
            table_name: Name of the table to store chunks
        """
        self.vector_dim = vector_dim
        self.table_name = table_name
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )

    def initialize(self) -> None:
        """Initialize database schema and enable pgvector extension."""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create chunks table with consistent schema
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    {self.TABLE_SCHEMA % self.vector_dim}
                );
            """)

            # Create vector similarity search index
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} 
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
                (embedding.tolist(), chunk["text"], chunk["metadata"])
                for chunk, embedding in zip(chunks, embeddings)
            ]
            
            execute_values(
                cur,
                f"""
                INSERT INTO {self.table_name} (embedding, text, metadata)
                VALUES %s
                """,
                values,
                template="(%s::vector, %s, %s::jsonb)"
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
                f"""
                SELECT text, metadata, date, 1 - (embedding <=> %s) as similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (query_embedding.tolist(), query_embedding.tolist(), limit)
            )
            
            results = []
            for text, metadata, date, similarity in cur.fetchall():
                results.append({
                    "text": text,
                    "metadata": metadata,
                    "date": date.isoformat(),
                    "similarity": float(similarity)
                })
                
            return results


class SupabaseDatabase(BaseDatabase):
    """Supabase database provider."""

    def __init__(
        self,
        url: str = None,
        key: str = None,
        table_name: str = "chunks",
        vector_dim: int = 1536
    ):
        """Initialize Supabase client.

        Args:
            url: Supabase project URL
            key: Supabase API key
            table_name: Name of the table to store chunks
            vector_dim: Dimension of vectors to store
        """
        if not url or not key:
            raise ValueError("Supabase URL and key must be provided")
            
        self.client = create_client(url, key)
        self.table_name = table_name
        self.vector_dim = vector_dim

    def initialize(self) -> None:
        """Initialize database schema using Supabase SQL editor.

        Note: For Supabase, you need to create the table and enable pgvector
        through the dashboard SQL editor with the following SQL:

        create extension if not exists vector;

        create table if not exists chunks (
            id bigint primary key generated always as identity,
            embedding vector(1536),
            text text not null,
            metadata jsonb,
            date timestamptz default now()
        );

        create index on chunks using ivfflat (embedding vector_cosine_ops)
        with (lists = 100);
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
                "embedding": embedding.tolist(),
                "text": chunk["text"],
                "metadata": chunk["metadata"]
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
                "date": item["date"],
                "similarity": item["similarity"]
            })
            
        return results 