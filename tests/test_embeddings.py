"""Test embedding providers."""
import os
from pathlib import Path

import numpy as np

from vector_search.config import Config
from vector_search.embeddings import OllamaEmbedding, OpenAIEmbedding


def test_ollama_embedding():
    """Test Ollama embedding provider."""
    config = Config("config.yaml")
    embedding = OllamaEmbedding(config, model="bge-m3:latest")
    
    # Test single text
    text = "This is a test document for embedding generation."
    result = embedding.embed(text)
    print(f"Single text embedding shape: {result.shape}")
    
    # Test multiple texts
    texts = [
        "First test document",
        "Second test document with different content",
        "Third document for testing embeddings"
    ]
    results = embedding.embed(texts)
    print(f"Multiple texts embedding shape: {results.shape}")


def test_openai_embedding():
    """Test OpenAI embedding provider."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping OpenAI test - API key not set")
        return
        
    config = Config("config.yaml")
    embedding = OpenAIEmbedding(config)
    
    # Test single text
    text = "This is a test document for OpenAI embedding."
    result = embedding.embed(text)
    print(f"Single text embedding shape: {result.shape}")
    
    # Test multiple texts
    texts = [
        "First OpenAI test",
        "Second OpenAI test with different content",
        "Third document for OpenAI embeddings"
    ]
    results = embedding.embed(texts)
    print(f"Multiple texts embedding shape: {results.shape}")


def test_custom_embedding():
    """Test custom embedding function."""
    def mock_embed(texts):
        """Mock embedding function that returns random vectors."""
        if isinstance(texts, str):
            texts = [texts]
        return np.random.randn(len(texts), 384)  # 384-dimensional embeddings
    
    from vector_search.embeddings import CustomEmbedding
    
    config = Config("config.yaml")
    embedding = CustomEmbedding(config, mock_embed)
    
    # Test single text
    text = "This is a test for custom embedding."
    result = embedding.embed(text)
    print(f"Single text custom embedding shape: {result.shape}")
    
    # Test multiple texts
    texts = [
        "First custom test",
        "Second custom test",
        "Third custom test"
    ]
    results = embedding.embed(texts)
    print(f"Multiple texts custom embedding shape: {results.shape}")


if __name__ == "__main__":
    print("\nTesting Ollama embeddings:")
    test_ollama_embedding()
    
    # print("\nTesting OpenAI embeddings:")
    # test_openai_embedding()
    
    # print("\nTesting custom embeddings:")
    # test_custom_embedding() 