"""Tag generation for text chunks using LangChain."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
import os

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser


class BaseTagGenerator(ABC):
    """Base class for tag generation."""

    def __init__(self, max_tags: int = 3, temperature: float = 0.3):
        """Initialize base tag generator.

        Args:
            max_tags: Maximum number of tags to generate
            temperature: Model temperature (0.0 to 1.0)
        """
        self.max_tags = max_tags
        self.temperature = temperature
        self.parser = CommaSeparatedListOutputParser()

    @abstractmethod
    def _setup_model(self):
        """Set up the language model."""
        pass

    @abstractmethod
    def _setup_prompt(self):
        """Set up the prompt template."""
        pass

    def generate_tags(self, chunks: List[Dict]) -> List[Dict]:
        """Generate tags for chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of chunks with tags added to metadata
        """
        tagged_chunks = []
        
        for chunk in chunks:
            # Format the prompt
            chain = self.prompt | self.llm | self.parser
            
            # Generate tags
            try:
                tags = chain.invoke({
                    "text": chunk["text"],
                    "max_tags": self.max_tags
                })
                
                # Create new chunk with tags
                tagged_chunk = chunk.copy()
                tagged_chunk["metadata"] = {
                    **chunk["metadata"],
                    "tags": tags
                }
                tagged_chunks.append(tagged_chunk)
                
            except Exception as e:
                print(f"Error generating tags for chunk: {str(e)}")
                tagged_chunks.append(chunk)  # Keep original chunk on error
        
        return tagged_chunks


class OllamaTagGenerator(BaseTagGenerator):
    """Generate tags using Ollama models."""

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        max_tags: int = 3,
        temperature: float = 0.3,
        base_url: Optional[str] = None
    ):
        """Initialize Ollama tag generator.

        Args:
            model_name: Name of the Ollama model to use
            max_tags: Maximum number of tags to generate
            temperature: Model temperature (0.0 to 1.0)
            base_url: Base URL for Ollama API
        """
        super().__init__(max_tags, temperature)
        self.model_name = model_name
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._setup_model()
        self._setup_prompt()

    def _setup_model(self):
        """Set up the Ollama model."""
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature
        )

    def _setup_prompt(self):
        """Set up the prompt template."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a precise tag generator. Generate exactly {max_tags} relevant "
                "tags for the given text. Tags should be specific, concise, and "
                "relevant to the main topics and concepts in the text. Separate tags "
                "with commas."
            )),
            ("user", "Text: {text}\nGenerate {max_tags} tags:")
        ])


class OpenAITagGenerator(BaseTagGenerator):
    """Generate tags using OpenAI models."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        max_tags: int = 3,
        temperature: float = 0.3,
        api_key: Optional[str] = None
    ):
        """Initialize OpenAI tag generator.

        Args:
            model_name: Name of the OpenAI model to use
            max_tags: Maximum number of tags to generate
            temperature: Model temperature (0.0 to 1.0)
            api_key: OpenAI API key
        """
        super().__init__(max_tags, temperature)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self._setup_model()
        self._setup_prompt()

    def _setup_model(self):
        """Set up the OpenAI model."""
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature
        )

    def _setup_prompt(self):
        """Set up the prompt template."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a precise tag generator. Generate exactly {max_tags} relevant "
                "tags for the given text. Tags should be specific, concise, and "
                "relevant to the main topics and concepts in the text. Separate tags "
                "with commas."
            )),
            ("user", "Text: {text}\nGenerate {max_tags} tags:")
        ])


class AzureOpenAITagGenerator(BaseTagGenerator):
    """Generate tags using Azure OpenAI models."""

    def __init__(
        self,
        deployment: str = "text-davinci-003",
        max_tags: int = 3,
        temperature: float = 0.3,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint: Optional[str] = None
    ):
        """Initialize Azure OpenAI tag generator.

        Args:
            deployment: Azure OpenAI deployment name
            max_tags: Maximum number of tags to generate
            temperature: Model temperature (0.0 to 1.0)
            api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
            endpoint: Azure OpenAI endpoint
        """
        super().__init__(max_tags, temperature)
        self.deployment = deployment
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not all([self.api_key, self.api_version, self.endpoint]):
            raise ValueError("Azure OpenAI credentials not fully provided")
        
        self._setup_model()
        self._setup_prompt()

    def _setup_model(self):
        """Set up the Azure OpenAI model."""
        self.llm = AzureChatOpenAI(
            deployment_name=self.deployment,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            temperature=self.temperature
        )

    def _setup_prompt(self):
        """Set up the prompt template."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a precise tag generator. Generate exactly {max_tags} relevant "
                "tags for the given text. Tags should be specific, concise, and "
                "relevant to the main topics and concepts in the text. Separate tags "
                "with commas."
            )),
            ("user", "Text: {text}\nGenerate {max_tags} tags:")
        ])


class OllamaPredefinedTagSelector(OllamaTagGenerator):
    """Select from predefined tags using Ollama."""

    def __init__(
        self,
        predefined_tags: Set[str],
        model_name: str = "llama2",
        max_tags: int = 3,
        temperature: float = 0.3,
        base_url: Optional[str] = None
    ):
        """Initialize Ollama predefined tag selector.

        Args:
            predefined_tags: Set of allowed tags
            model_name: Name of the Ollama model to use
            max_tags: Maximum number of tags to select
            temperature: Model temperature (0.0 to 1.0)
            base_url: Base URL for Ollama API
        """
        self.predefined_tags = predefined_tags
        super().__init__(model_name, max_tags, temperature, base_url)

    def _setup_prompt(self):
        """Set up the prompt template with predefined tags."""
        tags_list = ", ".join(sorted(self.predefined_tags))
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a precise tag selector. From the following list of tags:\n"
                f"{tags_list}\n\n"
                "Select exactly {max_tags} most relevant tags for the given text. "
                "Only use tags from the provided list. Separate tags with commas."
            )),
            ("user", "Text: {text}\nSelect {max_tags} tags:")
        ])

    def generate_tags(self, chunks: List[Dict]) -> List[Dict]:
        """Select relevant tags from predefined list.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of chunks with selected tags added to metadata
        """
        tagged_chunks = []
        
        for chunk in chunks:
            chain = self.prompt | self.llm | self.parser
            
            try:
                tags = chain.invoke({
                    "text": chunk["text"],
                    "max_tags": self.max_tags
                })
                
                # Validate tags are from predefined list
                valid_tags = [tag for tag in tags if tag in self.predefined_tags]
                
                tagged_chunk = chunk.copy()
                tagged_chunk["metadata"] = {
                    **chunk["metadata"],
                    "tags": valid_tags[:self.max_tags]
                }
                tagged_chunks.append(tagged_chunk)
                
            except Exception as e:
                print(f"Error selecting tags for chunk: {str(e)}")
                tagged_chunks.append(chunk)
        
        return tagged_chunks


class OpenAIPredefinedTagSelector(OpenAITagGenerator):
    """Select from predefined tags using OpenAI."""

    def __init__(
        self,
        predefined_tags: Set[str],
        model_name: str = "gpt-3.5-turbo",
        max_tags: int = 3,
        temperature: float = 0.3,
        api_key: Optional[str] = None
    ):
        """Initialize OpenAI predefined tag selector.

        Args:
            predefined_tags: Set of allowed tags
            model_name: Name of the OpenAI model to use
            max_tags: Maximum number of tags to select
            temperature: Model temperature (0.0 to 1.0)
            api_key: OpenAI API key
        """
        self.predefined_tags = predefined_tags
        super().__init__(model_name, max_tags, temperature, api_key)

    def _setup_prompt(self):
        """Set up the prompt template with predefined tags."""
        tags_list = ", ".join(sorted(self.predefined_tags))
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a precise tag selector. From the following list of tags:\n"
                f"{tags_list}\n\n"
                "Select exactly {max_tags} most relevant tags for the given text. "
                "Only use tags from the provided list. Separate tags with commas."
            )),
            ("user", "Text: {text}\nSelect {max_tags} tags:")
        ])

    def generate_tags(self, chunks: List[Dict]) -> List[Dict]:
        """Select relevant tags from predefined list.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of chunks with selected tags added to metadata
        """
        tagged_chunks = []
        
        for chunk in chunks:
            chain = self.prompt | self.llm | self.parser
            
            try:
                tags = chain.invoke({
                    "text": chunk["text"],
                    "max_tags": self.max_tags
                })
                
                # Validate tags are from predefined list
                valid_tags = [tag for tag in tags if tag in self.predefined_tags]
                
                tagged_chunk = chunk.copy()
                tagged_chunk["metadata"] = {
                    **chunk["metadata"],
                    "tags": valid_tags[:self.max_tags]
                }
                tagged_chunks.append(tagged_chunk)
                
            except Exception as e:
                print(f"Error selecting tags for chunk: {str(e)}")
                tagged_chunks.append(chunk)
        
        return tagged_chunks


class AzureOpenAIPredefinedTagSelector(AzureOpenAITagGenerator):
    """Select from predefined tags using Azure OpenAI."""

    def __init__(
        self,
        predefined_tags: Set[str],
        deployment: str = "text-davinci-003",
        max_tags: int = 3,
        temperature: float = 0.3,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint: Optional[str] = None
    ):
        """Initialize Azure OpenAI predefined tag selector.

        Args:
            predefined_tags: Set of allowed tags
            deployment: Azure OpenAI deployment name
            max_tags: Maximum number of tags to select
            temperature: Model temperature (0.0 to 1.0)
            api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
            endpoint: Azure OpenAI endpoint
        """
        self.predefined_tags = predefined_tags
        super().__init__(deployment, max_tags, temperature, api_key, api_version, endpoint)

    def _setup_prompt(self):
        """Set up the prompt template with predefined tags."""
        tags_list = ", ".join(sorted(self.predefined_tags))
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a precise tag selector. From the following list of tags:\n"
                f"{tags_list}\n\n"
                "Select exactly {max_tags} most relevant tags for the given text. "
                "Only use tags from the provided list. Separate tags with commas."
            )),
            ("user", "Text: {text}\nSelect {max_tags} tags:")
        ])

    def generate_tags(self, chunks: List[Dict]) -> List[Dict]:
        """Select relevant tags from predefined list.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of chunks with selected tags added to metadata
        """
        tagged_chunks = []
        
        for chunk in chunks:
            chain = self.prompt | self.llm | self.parser
            
            try:
                tags = chain.invoke({
                    "text": chunk["text"],
                    "max_tags": self.max_tags
                })
                
                # Validate tags are from predefined list
                valid_tags = [tag for tag in tags if tag in self.predefined_tags]
                
                tagged_chunk = chunk.copy()
                tagged_chunk["metadata"] = {
                    **chunk["metadata"],
                    "tags": valid_tags[:self.max_tags]
                }
                tagged_chunks.append(tagged_chunk)
                
            except Exception as e:
                print(f"Error selecting tags for chunk: {str(e)}")
                tagged_chunks.append(chunk)
        
        return tagged_chunks 