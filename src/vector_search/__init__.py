"""Vector search package for document processing and similarity search."""

from .augmentation import (AzureOpenAIAugmenter, BaseAugmenter, OllamaAugmenter,
                         OpenAIAugmenter)
from .chunker import BaseChunker, CharacterChunker, CustomChunker, WordChunker
from .database import BaseDatabase, PostgresDatabase, SupabaseDatabase
from .embeddings import (AzureOpenAIEmbedding, BaseEmbedding, CustomEmbedding,
                        OllamaEmbedding, OpenAIEmbedding)
from .sources import (AzureBlobSource, BaseSource, FileSource, FolderSource,
                     GoogleDriveSource)
from .vector_search import VectorSearch

__version__ = "0.1.0"

__all__ = [
    # Main class
    "VectorSearch",
    
    # Base classes
    "BaseSource",
    "BaseChunker",
    "BaseEmbedding",
    "BaseDatabase",
    "BaseAugmenter",
    
    
    # Source providers
    "FolderSource",
    "FileSource",
    "GoogleDriveSource",
    "AzureBlobSource",
    
    # Chunkers
    "WordChunker",
    "CharacterChunker",
    "CustomChunker",
    
    # Embedding providers
    "OllamaEmbedding",
    "OpenAIEmbedding",
    "AzureOpenAIEmbedding",
    "CustomEmbedding",
    
    # Database providers
    "PostgresDatabase",
    "SupabaseDatabase",
    
    # Augmenters
    "OllamaAugmenter",
    "OpenAIAugmenter",
    "AzureOpenAIAugmenter"
]

def main() -> None:
    print("Hello from vector-search!")
