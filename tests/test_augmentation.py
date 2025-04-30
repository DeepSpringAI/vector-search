"""Test text augmentation strategies."""
from pathlib import Path
import os
from dotenv import load_dotenv

from vector_search.augmentation import (
    OllamaAugmenter,
    OpenAIAugmenter,
    AzureOpenAIAugmenter
)


def test_ollama_augmenter():
    """Test Ollama-based augmentation."""
    # Initialize with environment variable OLLAMA_BASE_URL
    augmenter = OllamaAugmenter(model_name="llama3.1:8b")
    
    # Test document chunks
    chunks = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
            "metadata": {
                "source": "test_doc",
                "format": "txt"
            },
            "chunk_index": 0
        },
        {
            "text": "Python is a popular programming language known for its simplicity and readability.",
            "metadata": {
                "source": "test_doc",
                "format": "txt"
            },
            "chunk_index": 1
        }
    ]
    
    augmented_chunks = augmenter.augment(chunks)
    print("\nOllama Augmenter Results:")
    for i, chunk in enumerate(augmented_chunks):
        print(f"\nChunk {i + 1}:")
        print(f"Original text: {chunk['text']}")
        if chunk['metadata'].get('augmented'):
            print(f"Original chunk index: {chunk['metadata']['original_chunk_index']}")


def test_openai_augmenter():
    """Test OpenAI-based augmentation."""
    # Initialize with environment variable OPENAI_API_KEY
    augmenter = OpenAIAugmenter(model_name="gpt-3.5-turbo")
    
    # Test document chunks
    chunks = [
        {
            "text": "Vector databases are specialized databases designed to store and search high-dimensional vectors.",
            "metadata": {
                "source": "test_doc",
                "format": "txt"
            },
            "chunk_index": 0
        },
        {
            "text": "Natural Language Processing (NLP) is a field of AI focused on enabling computers to understand human language.",
            "metadata": {
                "source": "test_doc",
                "format": "txt"
            },
            "chunk_index": 1
        }
    ]
    
    augmented_chunks = augmenter.augment(chunks)
    print("\nOpenAI Augmenter Results:")
    for i, chunk in enumerate(augmented_chunks):
        print(f"\nChunk {i + 1}:")
        print(f"Original text: {chunk['text']}")
        if chunk['metadata'].get('augmented'):
            print(f"Original chunk index: {chunk['metadata']['original_chunk_index']}")


def test_azure_openai_augmenter():
    """Test Azure OpenAI-based augmentation."""
    # Initialize with environment variables
    augmenter = AzureOpenAIAugmenter()
    
    # Test document chunks
    chunks = [
        {
            "text": "Deep learning is a type of machine learning based on artificial neural networks.",
            "metadata": {
                "source": "test_doc",
                "format": "txt"
            },
            "chunk_index": 0
        },
        {
            "text": "Transformers have revolutionized natural language processing with their attention mechanism.",
            "metadata": {
                "source": "test_doc",
                "format": "txt"
            },
            "chunk_index": 1
        }
    ]
    
    augmented_chunks = augmenter.augment(chunks)
    print("\nAzure OpenAI Augmenter Results:")
    for i, chunk in enumerate(augmented_chunks):
        print(f"\nChunk {i + 1}:")
        print(f"Original text: {chunk['text']}")
        if chunk['metadata'].get('augmented'):
            print(f"Original chunk index: {chunk['metadata']['original_chunk_index']}")


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    print("Testing Ollama Augmenter:")
    test_ollama_augmenter()
    
    # if os.getenv("OPENAI_API_KEY"):
    #     print("\nTesting OpenAI Augmenter:")
    #     test_openai_augmenter()
    # else:
    #     print("\nSkipping OpenAI Augmenter - API key not set")
    
    # if os.getenv("AZURE_OPENAI_API_KEY"):
    #     print("\nTesting Azure OpenAI Augmenter:")
    #     test_azure_openai_augmenter()
    # else:
    #     print("\nSkipping Azure OpenAI Augmenter - API key not set") 