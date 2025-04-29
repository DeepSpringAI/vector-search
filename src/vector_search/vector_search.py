"""Main vector search implementation."""
from typing import Dict, List, Optional, Type, Union

import numpy as np
import os
from dotenv import load_dotenv

from .augmentation import (AzureOpenAIAugmenter, BaseAugmenter, OllamaAugmenter,
                         OpenAIAugmenter)
from .chunker import BaseChunker, CharacterChunker, CustomChunker, WordChunker
from .database import BaseDatabase, PostgresDatabase, SupabaseDatabase
from .embeddings import (AzureOpenAIEmbedding, BaseEmbedding, CustomEmbedding,
                        OllamaEmbedding, OpenAIEmbedding)
from .sources import (AzureBlobSource, BaseSource, FileSource, FolderSource,
                     GoogleDriveSource)


class VectorSearch:
    """Main vector search implementation."""

    def __init__(
        self,
        env_path: str = ".env",
        source_type: str = "folder",
        chunker_type: str = "word",
        embedding_type: str = "ollama",
        database_type: str = "postgres",
        augment: bool = False,
        augmenter_type: Optional[str] = None
    ):
        """Initialize vector search with components.

        Args:
            env_path: Path to .env file
            source_type: Type of input source ("folder", "file", "google_drive", "azure_blob")
            chunker_type: Type of text chunker ("word", "character")
            embedding_type: Type of embedding provider ("ollama", "openai", "azure_openai")
            database_type: Type of vector database ("postgres", "supabase")
            augment: Whether to augment chunks
            augmenter_type: Type of chunk augmenter ("ollama", "openai", "azure_openai")
        """
        # Load environment variables
        load_dotenv(env_path)
        
        # Initialize source
        source_map: Dict[str, Type[BaseSource]] = {
            "folder": FolderSource,
            "file": FileSource,
            "google_drive": GoogleDriveSource,
            "azure_blob": AzureBlobSource
        }
        self.source = source_map[source_type]()
        
        # Initialize chunker with chunk size and overlap from env
        chunker_map: Dict[str, Type[BaseChunker]] = {
            "word": WordChunker,
            "character": CharacterChunker
        }
        self.chunker = chunker_map[chunker_type](
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        
        # Initialize embedding provider
        embedding_map: Dict[str, Type[BaseEmbedding]] = {
            "ollama": OllamaEmbedding,
            "openai": OpenAIEmbedding,
            "azure_openai": AzureOpenAIEmbedding
        }
        self.embedding = embedding_map[embedding_type]()
        
        # Initialize database
        if database_type == "postgres":
            self.database = PostgresDatabase(
                dbname=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                host=os.getenv("POSTGRES_HOST"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                vector_dim=int(os.getenv("VECTOR_DIM", "1536"))
            )
        elif database_type == "supabase":
            self.database = SupabaseDatabase(
                url=os.getenv("SUPABASE_URL"),
                key=os.getenv("SUPABASE_KEY"),
                table_name=os.getenv("SUPABASE_TABLE", "chunks")
            )
        else:
            raise ValueError(f"Unsupported database type: {database_type}")

        # Initialize database
        self.database.initialize()
        
        # Initialize augmenter if enabled
        self.augment = augment
        if augment and augmenter_type:
            augmenter_map: Dict[str, Type[BaseAugmenter]] = {
                "ollama": OllamaAugmenter,
                "openai": OpenAIAugmenter,
                "azure_openai": AzureOpenAIAugmenter
            }
            self.augmenter = augmenter_map[augmenter_type]()
        else:
            self.augmenter = None

    def process_source(self, source_path: str) -> None:
        """Process documents from source and store in database.

        Args:
            source_path: Path or identifier for the source
        """
        # Load and chunk documents
        chunks = []
        for doc in self.source.load_documents(source_path):
            doc_chunks = self.chunker.chunk_text(doc["text"], doc["metadata"])
            chunks.extend(doc_chunks)
        
        # Augment chunks if enabled
        if self.augment and self.augmenter:
            chunks = self.augmenter.augment(chunks)
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding.embed(texts)
        
        # Store in database
        self.database.store_embeddings(chunks, embeddings)

    def search(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """Search for similar chunks.

        Args:
            query: Search query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of chunk dictionaries sorted by similarity
        """
        # Generate query embedding
        query_embedding = self.embedding.embed(query)
        
        # Search database
        results = self.database.search(query_embedding, limit)
        
        # Filter by similarity threshold
        if min_similarity > 0:
            results = [
                result for result in results
                if result["similarity"] >= min_similarity
            ]
        
        return results

    def add_custom_chunker(self, chunk_strategy: callable) -> None:
        """Add custom chunking strategy.

        Args:
            chunk_strategy: Custom chunking function
        """
        self.chunker = CustomChunker(
            chunk_strategy=chunk_strategy,
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )

    def add_custom_embedding(self, embed_fn: callable) -> None:
        """Add custom embedding function.

        Args:
            embed_fn: Custom embedding function
        """
        self.embedding = CustomEmbedding(embed_fn) 