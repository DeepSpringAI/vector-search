"""Test tag generation functionality."""
from vector_search.tags import (
    OllamaTagGenerator,
    OpenAITagGenerator,
    AzureOpenAITagGenerator,
    OllamaPredefinedTagSelector,
    OpenAIPredefinedTagSelector,
    AzureOpenAIPredefinedTagSelector
)
import os


def test_ollama_tag_generator():
    """Test Ollama-based tag generation."""
    generator = OllamaTagGenerator(
        model_name="llama3.1:8b",
        max_tags=3
    )
    
    # Test chunks
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
    
    # Generate tags
    tagged_chunks = generator.generate_tags(chunks)
    
    print("\nOllama Tag Generator Results:")
    for chunk in tagged_chunks:
        print(f"\nText: {chunk['text']}")
        print(f"Tags: {chunk['metadata'].get('tags', [])}")


def test_openai_tag_generator():
    """Test OpenAI-based tag generation."""
    generator = OpenAITagGenerator(
        model_name="gpt-3.5-turbo",
        max_tags=3
    )
    
    # Test chunks
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
    
    # Generate tags
    tagged_chunks = generator.generate_tags(chunks)
    
    print("\nOpenAI Tag Generator Results:")
    for chunk in tagged_chunks:
        print(f"\nText: {chunk['text']}")
        print(f"Tags: {chunk['metadata'].get('tags', [])}")


def test_azure_openai_tag_generator():
    """Test Azure OpenAI-based tag generation."""
    generator = AzureOpenAITagGenerator(
        deployment="your-deployment",
        max_tags=3
    )
    
    # Test chunks
    chunks = [
        {
            "text": "Deep learning models have revolutionized computer vision and natural language processing tasks.",
            "metadata": {
                "source": "test_doc",
                "format": "txt"
            },
            "chunk_index": 0
        },
        {
            "text": "Transformers have become the foundation for modern language models and text processing systems.",
            "metadata": {
                "source": "test_doc",
                "format": "txt"
            },
            "chunk_index": 1
        }
    ]
    
    # Generate tags
    tagged_chunks = generator.generate_tags(chunks)
    
    print("\nAzure OpenAI Tag Generator Results:")
    for chunk in tagged_chunks:
        print(f"\nText: {chunk['text']}")
        print(f"Tags: {chunk['metadata'].get('tags', [])}")


def test_ollama_predefined_tags():
    """Test Ollama-based predefined tag selection."""
    predefined_tags = {
        "machine learning",
        "artificial intelligence",
        "programming",
        "python",
        "data science",
        "software development",
        "algorithms",
        "deep learning",
        "web development",
        "database"
    }
    
    selector = OllamaPredefinedTagSelector(
        predefined_tags=predefined_tags,
        model_name="llama3.1:8b",
        max_tags=3
    )
    
    # Test chunks
    chunks = [
        {
            "text": "Deep learning models have revolutionized computer vision and natural language processing tasks.",
            "metadata": {
                "source": "test_doc",
                "format": "txt"
            },
            "chunk_index": 0
        },
        {
            "text": "Web developers often use Python frameworks like Django and Flask for building web applications.",
            "metadata": {
                "source": "test_doc",
                "format": "txt"
            },
            "chunk_index": 1
        }
    ]
    
    # Select tags
    tagged_chunks = selector.generate_tags(chunks)
    
    print("\nOllama Predefined Tag Selector Results:")
    for chunk in tagged_chunks:
        print(f"\nText: {chunk['text']}")
        print(f"Tags: {chunk['metadata'].get('tags', [])}")


if __name__ == "__main__":
    print("Testing Tag Generators:")
    
    print("\n1. Testing Ollama Tag Generator")
    test_ollama_tag_generator()
    
    # if os.getenv("OPENAI_API_KEY"):
    #     print("\n2. Testing OpenAI Tag Generator")
    #     test_openai_tag_generator()
    # else:
    #     print("\n2. Skipping OpenAI Tag Generator - API key not set")
    
    # if os.getenv("AZURE_OPENAI_API_KEY"):
    #     print("\n3. Testing Azure OpenAI Tag Generator")
    #     test_azure_openai_tag_generator()
    # else:
    #     print("\n3. Skipping Azure OpenAI Tag Generator - API key not set")
    
    print("\n4. Testing Ollama Predefined Tag Selector")
    test_ollama_predefined_tags() 