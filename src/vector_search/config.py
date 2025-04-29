"""Configuration management for vector search."""
import os
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class Config:
    """Configuration for vector search components."""

    # Database configuration
    db_host: str = None
    db_port: int = None
    db_name: str = None
    db_user: str = None
    db_password: str = None
    vector_dim: int = 384

    # Embedding configuration
    openai_api_key: str = None
    azure_api_key: str = None
    azure_api_version: str = None
    azure_endpoint: str = None
    azure_deployment: str = None
    ollama_base_url: str = None

    # Source configuration
    google_credentials: str = None
    azure_connection_string: str = None
    azure_container: str = None

    # Chunking configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from file and environment variables.

        Args:
            config_path: Path to YAML configuration file
        """
        # Load from config file if it exists
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if config:
                for key, value in config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

        # Override with environment variables if set
        env_map = {
            "db_host": "DB_HOST",
            "db_port": "DB_PORT",
            "db_name": "DB_NAME",
            "db_user": "DB_USER",
            "db_password": "DB_PASSWORD",
            "vector_dim": "VECTOR_DIM",
            "openai_api_key": "OPENAI_API_KEY",
            "azure_api_key": "AZURE_OPENAI_API_KEY",
            "azure_api_version": "AZURE_OPENAI_API_VERSION",
            "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
            "azure_deployment": "AZURE_OPENAI_DEPLOYMENT",
            "ollama_base_url": "OLLAMA_BASE_URL",
            "google_credentials": "GOOGLE_APPLICATION_CREDENTIALS",
            "azure_connection_string": "AZURE_STORAGE_CONNECTION_STRING",
            "azure_container": "AZURE_STORAGE_CONTAINER",
            "chunk_size": "CHUNK_SIZE",
            "chunk_overlap": "CHUNK_OVERLAP"
        }

        for attr, env_var in env_map.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to int for numeric fields
                if attr in ["db_port", "vector_dim", "chunk_size", "chunk_overlap"]:
                    env_value = int(env_value)
                setattr(self, attr, env_value) 