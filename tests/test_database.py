"""Test database providers."""
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from vector_search.database import PostgresDatabase, SupabaseDatabase


def test_postgres_database():
    """Test PostgreSQL database provider."""
    load_dotenv()
    
    if not os.getenv("POSTGRES_DB"):
        print("\nSkipping PostgreSQL test - database credentials not set")
        return
        
    db = PostgresDatabase(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        vector_dim=int(os.getenv("VECTOR_DIM", "1536"))
    )
    
    # Initialize database
    print("\nInitializing PostgreSQL database...")
    db.initialize()
    
    # Test data
    chunks = [
        {
            "text": "This is the first test chunk about machine learning.",
            "metadata": {
                "source": "test1",
                "format": "txt"
            }
        },
        {
            "text": "This is the second test chunk about natural language processing.",
            "metadata": {
                "source": "test2",
                "format": "txt"
            }
        },
        {
            "text": "This is the third test chunk about vector search.",
            "metadata": {
                "source": "test3",
                "format": "txt"
            }
        }
    ]
    
    # Generate mock embeddings (384-dimensional)
    embeddings = np.random.randn(len(chunks), 384)
    
    # Store embeddings
    print("Storing test chunks and embeddings...")
    db.store_embeddings(chunks, embeddings)
    
    # Test search
    print("\nTesting similarity search:")
    query_embedding = np.random.randn(384)  # Random query vector
    results = db.search(query_embedding, limit=2)
    
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Text: {result['text']}")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Metadata: {result['metadata']}")


def test_supabase_database():
    """Test Supabase database provider."""
    load_dotenv()
    
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("\nSkipping Supabase test - credentials not set")
        return
        
    db = SupabaseDatabase(
        url=os.getenv("SUPABASE_URL"),
        key=os.getenv("SUPABASE_KEY"),
        table_name=os.getenv("SUPABASE_TABLE", "chunks")
    )
    
    # Initialize database (no-op for Supabase as it's managed through dashboard)
    print("\nInitializing Supabase database...")
    db.initialize()
    
    # Test data
    chunks = [
        {
            "text": "Testing Supabase vector search with first chunk.",
            "metadata": {
                "source": "supabase_test1",
                "format": "txt"
            }
        },
        {
            "text": "Second test chunk for Supabase database.",
            "metadata": {
                "source": "supabase_test2",
                "format": "txt"
            }
        }
    ]
    
    # Generate mock embeddings (384-dimensional)
    embeddings = np.random.randn(len(chunks), 1024)
    
    # Store embeddings
    print("Storing test chunks and embeddings in Supabase...")
    try:
        db.store_embeddings(chunks, embeddings)
        
        # # Test search
        # print("\nTesting Supabase similarity search:")
        # query_embedding = np.random.randn(384)  # Random query vector
        # results = db.search(query_embedding, limit=2)
        
        # for i, result in enumerate(results):
        #     print(f"\nResult {i + 1}:")
        #     print(f"Text: {result['text']}")
        #     print(f"Similarity: {result.get('similarity', 'N/A')}")
        #     print(f"Metadata: {result.get('metadata', {})}")
            
    except Exception as e:
        print(f"Error during Supabase operations: {e}")


if __name__ == "__main__":
    # print("Testing PostgreSQL database:")
    # test_postgres_database()
    
    print("\nTesting Supabase database:")
    test_supabase_database() 