"""Text chunking strategies for vector search."""
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union


class BaseChunker(ABC):
    """Base class for text chunking strategies."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize chunker.

        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Number of overlapping units between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Chunk text into smaller pieces.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to include with chunks

        Returns:
            List of dictionaries containing chunks and metadata
        """
        pass

    def _create_metadata(self, source: Union[str, Path], metadata: Optional[Dict] = None) -> Dict:
        """Create metadata for chunks.

        Args:
            source: Source of the text
            metadata: Additional metadata to include

        Returns:
            Dictionary containing metadata
        """
        base_metadata = {
            'source': str(Path(source).stem),
            'date': datetime.now().strftime('%m-%d-%Y'),
            'path': str(source)
        }
        
        if metadata:
            base_metadata.update(metadata)
            
        return base_metadata


class WordChunker(BaseChunker):
    """Chunk text based on word count."""

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Chunk text based on word count with overlap.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to include with chunks

        Returns:
            List of dictionaries containing chunks and metadata
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append({
                    'text': chunk,
                    'metadata': metadata or {},
                    'chunk_index': len(chunks)
                })
                
        return chunks


class CharacterChunker(BaseChunker):
    """Chunk text based on character count."""

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Chunk text based on character count with overlap.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to include with chunks

        Returns:
            List of dictionaries containing chunks and metadata
        """
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append({
                    'text': chunk,
                    'metadata': metadata or {},
                    'chunk_index': len(chunks)
                })
                
        return chunks


class CustomChunker(BaseChunker):
    """Custom chunking strategy."""

    def __init__(self, chunk_strategy: callable, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize custom chunker.

        Args:
            chunk_strategy: Custom function for chunking
            chunk_size: Size of each chunk
            chunk_overlap: Number of overlapping units between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        self.chunk_strategy = chunk_strategy

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Apply custom chunking strategy.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to include with chunks

        Returns:
            List of dictionaries containing chunks and metadata
        """
        chunks = self.chunk_strategy(text)
        return [
            {
                'text': chunk,
                'metadata': metadata or {},
                'chunk_index': i
            }
            for i, chunk in enumerate(chunks)
        ] 