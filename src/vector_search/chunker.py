"""Text chunking strategies for vector search."""
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import Config


class BaseChunker(ABC):
    """Base class for text chunking strategies."""

    def __init__(self, config: Config):
        """Initialize chunker with configuration.

        Args:
            config: Configuration instance
        """
        self.config = config

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

    def __init__(self, config: Config):
        """Initialize word chunker.

        Args:
            config: Configuration instance
        """
        super().__init__(config)
        self.chunk_size = config.chunk_size
        self.overlap = config.chunk_overlap

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
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
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

    def __init__(self, config: Config):
        """Initialize character chunker.

        Args:
            config: Configuration instance
        """
        super().__init__(config)
        self.chunk_size = config.chunk_size
        self.overlap = config.chunk_overlap

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Chunk text based on character count with overlap.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to include with chunks

        Returns:
            List of dictionaries containing chunks and metadata
        """
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
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

    def __init__(self, config: Config, chunk_strategy: callable):
        """Initialize custom chunker.

        Args:
            config: Configuration instance
            chunk_strategy: Custom function for chunking
        """
        super().__init__(config)
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