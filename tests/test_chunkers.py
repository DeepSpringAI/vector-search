"""Test text chunking strategies."""
from pathlib import Path

from vector_search.config import Config
from vector_search.chunker import WordChunker, CharacterChunker, CustomChunker


def test_word_chunker():
    """Test word-based chunking."""
    config = Config("config.yaml")
    chunker = WordChunker(config)
    
    # Test document
    text = """
    This is a test document for word-based chunking. We'll create multiple
    sentences to ensure we have enough content for meaningful chunks. The chunker
    should split this text into overlapping segments based on word count.
    
    This is a second paragraph to make the text longer. We want to test how the
    chunker handles multiple paragraphs and maintains context across chunk
    boundaries. The overlap should help maintain coherence between chunks.
    """
    
    metadata = {
        "source": "test_doc",
        "format": "txt"
    }
    
    chunks = chunker.chunk_text(text, metadata)
    print("\nWord Chunker Results:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"Text length (words): {len(chunk['text'].split())}")
        print(f"First 50 chars: {chunk['text'][:50]}...")
        print(f"Metadata: {chunk['metadata']}")


def test_character_chunker():
    """Test character-based chunking."""
    config = Config("config.yaml")
    chunker = CharacterChunker(config)
    
    # Test document
    text = """
    This is a test document for character-based chunking. Instead of counting
    words, this chunker will split the text based on character count. This might
    be more appropriate for certain types of documents or languages where word
    boundaries are not well defined.
    
    Let's add another paragraph to test how the character chunker handles longer
    texts and maintains proper chunk boundaries. The overlap settings should
    ensure that we don't cut words in the middle.
    """
    
    metadata = {
        "source": "test_doc",
        "format": "txt"
    }
    
    chunks = chunker.chunk_text(text, metadata)
    print("\nCharacter Chunker Results:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"Text length (chars): {len(chunk['text'])}")
        print(f"First 50 chars: {chunk['text'][:50]}...")
        print(f"Metadata: {chunk['metadata']}")


def test_custom_chunker():
    """Test custom chunking strategy."""
    def sentence_chunker(text: str) -> list:
        """Simple sentence-based chunking strategy."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    config = Config("config.yaml")
    chunker = CustomChunker(config, sentence_chunker)
    
    # Test document
    text = """
    This is the first sentence. This is the second sentence! How about a third
    sentence? Now we're on the fourth sentence. The fifth sentence is here.
    
    Let's add some more sentences! This is getting interesting? Finally, we'll
    end with this sentence.
    """
    
    metadata = {
        "source": "test_doc",
        "format": "txt"
    }
    
    chunks = chunker.chunk_text(text, metadata)
    print("\nCustom Chunker (Sentence-based) Results:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"Text: {chunk['text']}")
        print(f"Metadata: {chunk['metadata']}")


if __name__ == "__main__":
    print("Testing Word Chunker:")
    test_word_chunker()
    
    print("\nTesting Character Chunker:")
    test_character_chunker()
    
    print("\nTesting Custom Chunker:")
    test_custom_chunker() 