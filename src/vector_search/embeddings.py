"""Embedding providers for vector search."""
import os
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import requests
from openai import AzureOpenAI, OpenAI

from .config import Config


class BaseEmbedding(ABC):
    """Base class for embedding providers."""

    def __init__(self, config: Config):
        """Initialize embedding provider.

        Args:
            config: Configuration instance
        """
        self.config = config

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

    def __init__(self, config: Config, model: str = "bge-m3:latest"):
        """Initialize Ollama embedding provider.

        Args:
            config: Configuration instance
            model: Model name to use (default: bge-m3:latest)
        """
        super().__init__(config)
        base_url = config.ollama_base_url or "http://localhost:11434"
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

    def __init__(self, config: Config):
        """Initialize OpenAI embedding provider.

        Args:
            config: Configuration instance
        """
        super().__init__(config)
        self.client = OpenAI(api_key=config.openai_api_key)

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

    def __init__(self, config: Config):
        """Initialize Azure OpenAI embedding provider.

        Args:
            config: Configuration instance
        """
        super().__init__(config)
        self.client = AzureOpenAI(
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint
        )
        self.deployment = config.azure_deployment or "text-embedding-ada-002"

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

    def __init__(self, config: Config, embed_fn: callable):
        """Initialize custom embedding provider.

        Args:
            config: Configuration instance
            embed_fn: Custom embedding function
        """
        super().__init__(config)
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