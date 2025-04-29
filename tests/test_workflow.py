"""Test complete workflow from text files to database storage."""
import os
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv

from vector_search.chunker import WordChunker
from vector_search.embeddings import OllamaEmbedding
from vector_search.database import PostgresDatabase


def setup_sample_texts(folder_path: str) -> List[Path]:
    """Create sample text files for testing.

    Args:
        folder_path: Path to create sample files in

    Returns:
        List of created file paths
    """
    # Create test folder
    test_folder = Path(folder_path)
    test_folder.mkdir(exist_ok=True)
    
    # Sample documents with different topics
    documents = {
        "machine_learning.txt": """
        Machine learning is a subset of artificial intelligence that focuses on
        developing systems that can learn from and make decisions based on data.
        It has numerous applications in various fields, from computer vision to
        natural language processing. Deep learning, a subset of machine learning,
        uses neural networks with multiple layers to learn hierarchical
        representations of data.
        
        Common machine learning algorithms include:
        1. Linear Regression
        2. Decision Trees
        3. Random Forests
        4. Support Vector Machines
        5. Neural Networks
        """,
        
        "database_systems.txt": """
        Database management systems (DBMS) are software applications that enable
        users to store, organize, and manage large amounts of data efficiently.
        They provide mechanisms for data persistence, concurrent access, and
        transaction management. Modern databases support various data models,
        including relational, document-oriented, and graph-based structures.
        
        Key features of database systems:
        - ACID properties (Atomicity, Consistency, Isolation, Durability)
        - Query optimization
        - Index management
        - Security and access control
        - Backup and recovery
        """,
        
        "web_development.txt": """
        Web development encompasses the creation and maintenance of websites and
        web applications. It involves both frontend development (client-side) and
        backend development (server-side). Modern web development often uses
        frameworks and libraries to accelerate development and ensure best
        practices.
        
        Popular web technologies include:
        - HTML5, CSS3, JavaScript
        - React, Vue, Angular
        - Node.js, Django, Flask
        - RESTful APIs and GraphQL
        - Containerization with Docker
        """
    }
    
    # Create files
    file_paths = []
    for filename, content in documents.items():
        file_path = test_folder / filename
        with open(file_path, "w") as f:
            f.write(content)
        file_paths.append(file_path)
    
    return file_paths


def process_files(file_paths: List[Path]) -> None:
    """Process text files and store in database.

    Args:
        file_paths: List of text file paths to process
    """
    # Initialize components
    chunker = WordChunker(chunk_size=20, chunk_overlap=5)
    embedding_model = OllamaEmbedding(model="bge-m3:latest")
    # db = PostgresDatabase(
    #     dbname=os.getenv("POSTGRES_DB"),
    #     user=os.getenv("POSTGRES_USER"),
    #     password=os.getenv("POSTGRES_PASSWORD"),
    #     host=os.getenv("POSTGRES_HOST", "localhost"),
    #     port=int(os.getenv("POSTGRES_PORT", "5432")),
    #     vector_dim=int(os.getenv("VECTOR_DIM", "1536"))
    # )
    
    # # Initialize database
    # print("\nInitializing database...")
    # db.initialize()
    
    # Process each file
    for file_path in file_paths:
        print(f"\nProcessing {file_path.name}...")
        
        # Read file
        with open(file_path, "r") as f:
            text = f.read()
        
        # Create metadata
        metadata = {
            "source": file_path.stem,
            "format": "txt",
            "path": str(file_path)
        }
        
        # Chunk text
        print("Chunking text...")
        chunks = chunker.chunk_text(text, metadata)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_model.embed(texts)
        print(f"Generated embeddings of shape {embeddings.shape}")
        
        # # Store in database
        # print("Storing in database...")
        # db.store_embeddings(chunks, embeddings)
        
        # # Test search with last chunk
        # print("\nTesting search with last chunk as query...")
        # results = db.search(embeddings[-1], limit=2)
        
        # print("\nSearch results:")
        # for i, result in enumerate(results):
        #     print(f"\nResult {i + 1}:")
        #     print(f"Text: {result['text'][:100]}...")
        #     print(f"Source: {result['metadata']['source']}")
        #     print(f"Similarity: {result['similarity']:.3f}")


def cleanup(folder_path: str) -> None:
    """Clean up test files.

    Args:
        folder_path: Path to clean up
    """
    folder = Path(folder_path)
    if folder.exists():
        for file in folder.glob("*"):
            file.unlink()
        folder.rmdir()


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Setup
    TEST_FOLDER = "test_samples"
    print(f"\nCreating sample files in {TEST_FOLDER}/...")
    file_paths = setup_sample_texts(TEST_FOLDER)
    
    try:
        # Process files
        process_files(file_paths)
    finally:
        # Cleanup
        print(f"\nCleaning up {TEST_FOLDER}/...")
        cleanup(TEST_FOLDER) 