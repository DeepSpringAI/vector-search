"""Text augmentation for vector search chunks."""
import os
from abc import ABC, abstractmethod
from typing import Dict, List

from openai import AzureOpenAI, OpenAI


class BaseAugmenter(ABC):
    """Base class for text augmentation."""

    @abstractmethod
    def augment(self, chunks: List[Dict]) -> List[Dict]:
        """Augment text chunks with rewritten versions.

        Args:
            chunks: List of chunk dictionaries containing text and metadata

        Returns:
            List of augmented chunk dictionaries
        """
        pass


class OllamaAugmenter(BaseAugmenter):
    """Ollama-based text augmentation."""

    def __init__(self, model_name: str = "llama2", base_url: str = None):
        """Initialize Ollama augmenter.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def augment(self, chunks: List[Dict]) -> List[Dict]:
        """Augment chunks using Ollama.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of augmented chunks
        """
        import requests

        augmented_chunks = []
        
        for chunk in chunks:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at rewriting text. Only respond with the rewritten text, no explanations or additional text."
                },
                {
                    "role": "user",
                    "content": f"Rewrite this text in a different way while preserving its core meaning: {chunk['text']}"
                }
            ]
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            augmented_text = response.json()["message"]["content"].strip()
            
            # Keep original chunk
            augmented_chunks.append(chunk)
            
            # Add augmented version
            augmented_chunk = chunk.copy()
            augmented_chunk["text"] = augmented_text
            augmented_chunk["metadata"] = {
                **chunk["metadata"],
                "augmented": True,
                "original_chunk_index": chunk["chunk_index"]
            }
            augmented_chunks.append(augmented_chunk)
            
        return augmented_chunks


class OpenAIAugmenter(BaseAugmenter):
    """OpenAI-based text augmentation."""

    def __init__(self, api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        """Initialize OpenAI augmenter.

        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI model to use
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def augment(self, chunks: List[Dict]) -> List[Dict]:
        """Augment chunks using OpenAI.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of augmented chunks
        """
        augmented_chunks = []
        
        for chunk in chunks:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at rewriting text while preserving "
                            "its core meaning and key information. Rewrite the "
                            "following text in a different way."
                        )
                    },
                    {"role": "user", "content": chunk["text"]}
                ]
            )
            
            augmented_text = response.choices[0].message.content.strip()
            
            # Keep original chunk
            augmented_chunks.append(chunk)
            
            # Add augmented version
            augmented_chunk = chunk.copy()
            augmented_chunk["text"] = augmented_text
            augmented_chunk["metadata"] = {
                **chunk["metadata"],
                "augmented": True,
                "original_chunk_index": chunk["chunk_index"]
            }
            augmented_chunks.append(augmented_chunk)
            
        return augmented_chunks


class AzureOpenAIAugmenter(BaseAugmenter):
    """Azure OpenAI-based text augmentation."""

    def __init__(
        self,
        api_key: str = None,
        api_version: str = None,
        endpoint: str = None,
        deployment: str = "text-davinci-003"
    ):
        """Initialize Azure OpenAI augmenter.

        Args:
            api_key: Azure OpenAI API key
            api_version: API version
            endpoint: Azure endpoint
            deployment: Deployment name to use
        """
        self.deployment = deployment
        self.client = AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def augment(self, chunks: List[Dict]) -> List[Dict]:
        """Augment chunks using Azure OpenAI.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of augmented chunks
        """
        augmented_chunks = []
        
        for chunk in chunks:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at rewriting text while preserving "
                            "its core meaning and key information. Rewrite the "
                            "following text in a different way."
                        )
                    },
                    {"role": "user", "content": chunk["text"]}
                ]
            )
            
            augmented_text = response.choices[0].message.content.strip()
            
            # Keep original chunk
            augmented_chunks.append(chunk)
            
            # Add augmented version
            augmented_chunk = chunk.copy()
            augmented_chunk["text"] = augmented_text
            augmented_chunk["metadata"] = {
                **chunk["metadata"],
                "augmented": True,
                "original_chunk_index": chunk["chunk_index"]
            }
            augmented_chunks.append(augmented_chunk)
            
        return augmented_chunks