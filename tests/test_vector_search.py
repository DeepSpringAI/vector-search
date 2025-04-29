"""Test the main VectorSearch implementation."""
import os
from pathlib import Path

from vector_search.vector_search import VectorSearch


def setup_test_data():
    """Create test data for vector search."""
    test_folder = Path("test_data")
    test_folder.mkdir(exist_ok=True)
    
    # Create sample documents
    documents = {
        "article1.txt": """
        Machine learning is a subset of artificial intelligence that focuses on
        developing systems that can learn from and make decisions based on data.
        It has numerous applications in various fields, from computer vision to
        natural language processing.
        """,
        
        "article2.txt": """
        Natural Language Processing (NLP) is a branch of AI that helps computers
        understand, interpret, and manipulate human language. It combines
        computational linguistics with statistical models to process text.
        """,
        
        "article3.txt": """
        Vector search, also known as semantic search, uses numerical
        representations of text (embeddings) to find similar content. It's more
        powerful than traditional keyword search because it can understand
        context and meaning.
        """
    }
    
    for filename, content in documents.items():
        with open(test_folder / filename, "w") as f:
            f.write(content)
            
    return test_folder


def test_basic_workflow():
    """Test basic vector search workflow."""
    # Create test data
    test_folder = setup_test_data()
    
    # Initialize vector search with Ollama (local) components
    vector_search = VectorSearch(
        source_type="folder",
        chunker_type="word",
        embedding_type="ollama",
        database_type="postgres",
        augment=False
    )
    
    print("\nProcessing documents...")
    vector_search.process_source(str(test_folder))
    
    # Test various queries
    queries = [
        "What is machine learning?",
        "Explain NLP",
        "How does semantic search work?",
        "What are embeddings used for?"
    ]
    
    print("\nSearch Results:")
    for query in queries:
        print(f"\nQuery: {query}")
        results = vector_search.search(query, limit=2)
        
        for i, result in enumerate(results):
            print(f"\nResult {i + 1}:")
            print(f"Text: {result['text'][:100]}...")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Source: {result['metadata'].get('source', 'unknown')}")
    
    # Cleanup
    for file in test_folder.glob("*"):
        file.unlink()
    test_folder.rmdir()


def test_augmented_search():
    """Test vector search with chunk augmentation."""
    if not os.getenv("OPENAI_API_KEY"):
        print("\nSkipping augmented search test - OpenAI API key not set")
        return
        
    # Create test data
    test_folder = setup_test_data()
    
    # Initialize vector search with OpenAI and augmentation
    vector_search = VectorSearch(
        source_type="folder",
        chunker_type="word",
        embedding_type="openai",
        database_type="postgres",
        augment=True,
        augmenter_type="openai"
    )
    
    print("\nProcessing documents with augmentation...")
    vector_search.process_source(str(test_folder))
    
    # Test queries
    queries = [
        "What are the main applications of ML?",
        "How do computers process human language?",
        "Compare keyword search with semantic search"
    ]
    
    print("\nAugmented Search Results:")
    for query in queries:
        print(f"\nQuery: {query}")
        results = vector_search.search(query, limit=2)
        
        for i, result in enumerate(results):
            print(f"\nResult {i + 1}:")
            print(f"Text: {result['text'][:100]}...")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Source: {result['metadata'].get('source', 'unknown')}")
            print(f"Augmented: {result['metadata'].get('augmented', False)}")
    
    # Cleanup
    for file in test_folder.glob("*"):
        file.unlink()
    test_folder.rmdir()


def test_custom_components():
    """Test vector search with custom components."""
    # Create test data
    test_folder = setup_test_data()
    
    # Initialize basic vector search
    vector_search = VectorSearch(
        source_type="folder",
        chunker_type="word",
        embedding_type="ollama"
    )
    
    # Add custom sentence chunker
    def sentence_chunker(text: str) -> list:
        """Simple sentence-based chunking."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    vector_search.add_custom_chunker(sentence_chunker)
    
    # Add custom mock embedding function
    def mock_embed(texts):
        """Mock embedding function."""
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        return np.random.randn(len(texts), 384)
    
    vector_search.add_custom_embedding(mock_embed)
    
    print("\nProcessing documents with custom components...")
    vector_search.process_source(str(test_folder))
    
    # Test search
    query = "How does machine learning work?"
    print(f"\nQuery: {query}")
    results = vector_search.search(query, limit=2)
    
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Text: {result['text']}")
        print(f"Similarity: {result['similarity']:.3f}")
    
    # Cleanup
    for file in test_folder.glob("*"):
        file.unlink()
    test_folder.rmdir()


if __name__ == "__main__":
    print("Testing basic vector search workflow:")
    test_basic_workflow()
    
    print("\nTesting augmented vector search:")
    test_augmented_search()
    
    print("\nTesting custom components:")
    test_custom_components() 