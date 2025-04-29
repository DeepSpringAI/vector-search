"""Embedding providers for vector search."""
import os
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import requests
from openai import AzureOpenAI, OpenAI


class BaseEmbedding(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input texts.

        Args:
            texts: Single text or list of texts to embed

        Returns:
            Numpy array of embeddings
        """
        pass


class OllamaEmbedding(BaseEmbedding):
    """Ollama embedding provider."""

    def __init__(self, model: str = "bge-m3:latest"):
        """Initialize Ollama embedding provider.

        Args:
            model: Model name to use (default: bge-m3:latest)
        """
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = f"{base_url}/api/embed"
        self.model = model

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Ollama.

        Args:
            texts: Single text or list of texts to embed

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        payload = {
            "model": self.model,
            "input": texts
        }

        try:
            response = requests.post(self.base_url, json=payload, verify=False)
            if response.status_code == 200:
                result = response.json()
                if 'embeddings' in result:
                    return np.array(result['embeddings'])
            return np.array([])
        except Exception:
            return np.array([])


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding provider."""

    def __init__(self):
        """Initialize OpenAI embedding provider."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI.

        Args:
            texts: Single text or list of texts to embed

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        return np.array([e.embedding for e in response.data])


class AzureOpenAIEmbedding(BaseEmbedding):
    """Azure OpenAI embedding provider."""

    def __init__(self):
        """Initialize Azure OpenAI embedding provider."""
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-ada-002")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Azure OpenAI.

        Args:
            texts: Single text or list of texts to embed

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model=self.deployment,
            input=texts
        )
        
        return np.array([e.embedding for e in response.data])


class CustomEmbedding(BaseEmbedding):
    """Custom embedding provider."""

    def __init__(self, embed_fn: callable):
        """Initialize custom embedding provider.

        Args:
            embed_fn: Custom embedding function
        """
        self.embed_fn = embed_fn

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using custom function.

        Args:
            texts: Single text or list of texts to embed

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        return np.array(self.embed_fn(texts))