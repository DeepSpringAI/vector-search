<div align="center">
  <img src="inc/logo.png" alt="Vector Search Logo" width="200"/>
</div>

# Vector Search

A flexible and modular vector search system for document processing, embedding generation, and similarity search.

## Features

- Multiple source providers (Folder, File, Google Drive, Azure Blob)
- Flexible text chunking strategies (Word-based, Character-based, Custom)
- Various embedding providers (Ollama, OpenAI, Azure OpenAI, Custom)
- Database options (PostgreSQL with pgvector, Supabase)
- Optional text augmentation for better search results

## Installation

### Development Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vector-search.git
cd vector-search
```

2. Create and activate a virtual environment using `uv`:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Using in Your Projects

To use vector-search in your own projects:

1. Create and activate a virtual environment for your project:
```bash
mkdir your-project
cd your-project
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install vector-search directly from the project path:
```bash
uv pip install '/path/to/vector-search'
```

Now you can import and use vector-search in your code:
```python
from vector_search import VectorSearch
```

### Environment Setup

Create a `.env` file in your project root:
```env
# Database Configuration
POSTGRES_DB=your_db_name
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
VECTOR_DIM=1536

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# OpenAI Configuration (if using OpenAI embeddings)
OPENAI_API_KEY=your_api_key

# Azure OpenAI Configuration (if using Azure OpenAI)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_DEPLOYMENT=your_deployment_name

# Ollama Configuration (if using Ollama)
OLLAMA_BASE_URL=http://localhost:11434

# Google Drive Configuration (if using Google Drive source)
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Azure Blob Configuration (if using Azure Blob source)
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_STORAGE_CONTAINER=your_container_name

# Supabase Configuration (if using Supabase)
SUPABASE_URL=your_project_url
SUPABASE_KEY=your_api_key
SUPABASE_TABLE=chunks
```

## Basic Usage

```python
from vector_search import VectorSearch

# Initialize with default components
vector_search = VectorSearch(
    source_type="folder",      # Use folder source
    chunker_type="word",       # Use word-based chunking
    embedding_type="ollama",   # Use Ollama for embeddings
    database_type="postgres",  # Use PostgreSQL database
    augment=False             # No text augmentation
)

# Process documents from a folder
vector_search.process_source("path/to/documents")

# Search for similar content
results = vector_search.search(
    query="What is machine learning?",
    limit=5,
    min_similarity=0.7
)

# Print results
for result in results:
    print(f"Text: {result['text']}")
    print(f"Similarity: {result['similarity']}")
    print(f"Source: {result['metadata']['source']}")
    print(f"Date: {result['date']}")
    print("---")
```

## Creating Custom Components

### 1. Custom Source Provider

Create a custom source by inheriting from `BaseSource`:

```python
from vector_search.sources import BaseSource
from typing import Dict, Iterator

class CustomSource(BaseSource):
    def __init__(self, supported_formats=None):
        super().__init__(supported_formats)
        # Add custom initialization

    def load_documents(self, source_path: str) -> Iterator[Dict]:
        # Implement document loading logic
        documents = []  # Your logic here
        
        for doc in documents:
            yield {
                "text": doc.content,
                "metadata": {
                    "source": doc.name,
                    "format": doc.format,
                    "custom_field": doc.custom_field
                }
            }
```

### 2. Custom Chunking Strategy

Create a custom chunker by inheriting from `BaseChunker`:

```python
from vector_search.chunker import BaseChunker
from typing import Dict, List

class CustomChunker(BaseChunker):
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        super().__init__(chunk_size, chunk_overlap)
        # Add custom initialization

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        # Implement chunking logic
        chunks = []  # Your logic here
        
        return [
            {
                "text": chunk,
                "metadata": metadata or {},
                "chunk_index": i
            }
            for i, chunk in enumerate(chunks)
        ]
```

### 3. Custom Embedding Provider

Create a custom embedding provider by inheriting from `BaseEmbedding`:

```python
from vector_search.embeddings import BaseEmbedding
import numpy as np
from typing import List, Union

class CustomEmbedding(BaseEmbedding):
    def __init__(self):
        # Add custom initialization
        pass

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        # Implement embedding generation logic
        embeddings = []  # Your logic here
        
        return np.array(embeddings)
```

### 4. Custom Database Provider

Create a custom database provider by inheriting from `BaseDatabase`:

```python
from vector_search.database import BaseDatabase
import numpy as np
from typing import Dict, List

class CustomDatabase(BaseDatabase):
    def __init__(self):
        # Add custom initialization
        pass

    def initialize(self) -> None:
        # Initialize database schema
        pass

    def store_embeddings(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        # Store chunks and embeddings
        pass

    def search(self, query_embedding: np.ndarray, limit: int = 5) -> List[Dict]:
        # Implement similarity search
        results = []  # Your logic here
        
        return results
```

### 5. Custom Augmenter

Create a custom augmenter by inheriting from `BaseAugmenter`:

```python
from vector_search.augmentation import BaseAugmenter
from typing import Dict, List

class CustomAugmenter(BaseAugmenter):
    def __init__(self):
        # Add custom initialization
        pass

    def augment(self, chunks: List[Dict]) -> List[Dict]:
        augmented_chunks = []
        
        for chunk in chunks:
            # Keep original chunk
            augmented_chunks.append(chunk)
            
            # Add augmented version
            augmented_text = self.generate_variation(chunk["text"])
            augmented_chunk = chunk.copy()
            augmented_chunk["text"] = augmented_text
            augmented_chunk["metadata"] = {
                **chunk["metadata"],
                "augmented": True,
                "original_chunk_index": chunk["chunk_index"]
            }
            augmented_chunks.append(augmented_chunk)
            
        return augmented_chunks

    def generate_variation(self, text: str) -> str:
        # Implement text variation generation
        return modified_text
```

## Using Custom Components

```python
# Initialize VectorSearch with custom components
vector_search = VectorSearch(
    source_type="folder",
    chunker_type="word",
    embedding_type="ollama",
    database_type="postgres"
)

# Add custom chunker
def sentence_chunker(text: str) -> list:
    import re
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

vector_search.add_custom_chunker(sentence_chunker)

# Add custom embedding function
def custom_embed(texts):
    # Your embedding logic here
    return embeddings

vector_search.add_custom_embedding(custom_embed)
```

## Database Schema

Both PostgreSQL and Supabase use the same table schema:

```sql
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    embedding vector(1536),
    text TEXT NOT NULL,
    metadata JSONB,
    date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

## Test Files and Examples

The `tests/` directory contains several test files that demonstrate how to use different components of the system:

### 1. Basic Workflow Test (`test_workflow.py`)
```python
# Complete example of processing files and storing in database
from vector_search.chunker import WordChunker
from vector_search.embeddings import OllamaEmbedding
from vector_search.database import PostgresDatabase

# Creates sample text files and processes them
test_folder = "test_samples"
file_paths = setup_sample_texts(test_folder)
process_files(file_paths)
```

### 2. Database Tests (`test_database.py`)
```python
# Test PostgreSQL and Supabase implementations
from vector_search.database import PostgresDatabase, SupabaseDatabase

# PostgreSQL example
db = PostgresDatabase(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    vector_dim=1536
)

# Store and search example data
db.initialize()
db.store_embeddings(chunks, embeddings)
results = db.search(query_embedding, limit=2)
```

### 3. Embedding Tests (`test_embeddings.py`)
```python
# Test different embedding providers
from vector_search.embeddings import OllamaEmbedding, OpenAIEmbedding

# Ollama example
embedding = OllamaEmbedding(model="bge-m3:latest")
result = embedding.embed("This is a test document")

# OpenAI example
embedding = OpenAIEmbedding()
result = embedding.embed("This is a test document")
```

### 4. Source Tests (`test_sources.py`)
```python
# Test different source providers
from vector_search.sources import FolderSource, FileSource, GoogleDriveSource

# Folder source example
source = FolderSource()
for doc in source.load_documents("path/to/folder"):
    print(f"Document: {doc['metadata']['source']}")
    print(f"Content: {doc['text'][:100]}...")
```

### 5. Chunker Tests (`test_chunkers.py`)
```python
# Test different chunking strategies
from vector_search.chunker import WordChunker, CharacterChunker

# Word chunker example
chunker = WordChunker(chunk_size=100, chunk_overlap=20)
chunks = chunker.chunk_text(text, metadata={"source": "test.txt"})
```

### 6. Vector Search Tests (`test_vector_search.py`)
```python
# Test the main VectorSearch implementation
from vector_search.vector_search import VectorSearch

# Basic workflow
vector_search = VectorSearch(
    source_type="folder",
    chunker_type="word",
    embedding_type="ollama",
    database_type="postgres"
)

# Process and search
vector_search.process_source("test_data")
results = vector_search.search("What is machine learning?", limit=5)
```

### Running the Tests

1. Make sure you have all dependencies installed and environment variables set
2. Run individual tests:
```bash
python -m pytest tests/test_database.py
python -m pytest tests/test_embeddings.py
python -m pytest tests/test_sources.py
python -m pytest tests/test_chunkers.py
python -m pytest tests/test_vector_search.py
python -m pytest tests/test_workflow.py
```

3. Run all tests:
```bash
python -m pytest tests/
```

The test files serve as both documentation and examples of how to use each component of the system. They demonstrate:
- Basic usage patterns
- Component initialization
- Error handling
- Integration between components
- Complete workflows

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Text Filtering

The source providers support custom text filtering through an optional `text_filter` parameter. This allows you to preprocess text content before it's processed further in the pipeline.

### Basic Usage

```python
from vector_search.sources import FolderSource, FileSource

# Create a simple lowercase filter
def lowercase_filter(text: str) -> str:
    return text.lower()

# Initialize source with the filter
source = FolderSource(text_filter=lowercase_filter)

# All documents loaded will have lowercase text
for doc in source.load_documents("path/to/folder"):
    print(doc["text"])  # Text will be lowercase
```

### Example Filters

1. Remove Punctuation:
```python
def remove_punctuation_filter(text: str) -> str:
    return "".join(char for char in text if char.isalnum() or char.isspace())

source = FileSource(text_filter=remove_punctuation_filter)
```

2. Clean Extra Whitespace:
```python
def clean_whitespace_filter(text: str) -> str:
    return " ".join(text.split())

source = FolderSource(text_filter=clean_whitespace_filter)
```

3. Custom Text Processing:
```python
def custom_filter(text: str) -> str:
    # Remove specific patterns
    text = text.replace("[REMOVE]", "")
    # Convert to lowercase
    text = text.lower()
    # Clean whitespace
    text = " ".join(text.split())
    return text

source = FileSource(text_filter=custom_filter)
```

### Filter Behavior

- The filter function receives the raw text content as a string
- It must return the processed text as a string
- The filter is applied before the text is added to the document dictionary
- Original metadata remains unchanged
- The filter is applied consistently across all supported file formats
- If no filter is provided, the text remains unchanged

### Supported Source Types

Text filters can be used with any source provider:
- `FolderSource`: Applied to all files in the folder
- `FileSource`: Applied to the single file
- `GoogleDriveSource`: Applied to Google Drive documents
- `AzureBlobSource`: Applied to Azure Blob Storage files

## Google Drive Integration

The system supports reading files directly from Google Drive, including native Google Workspace files (Docs, Sheets, Slides) which are automatically converted to Markdown format.

### Setting up Google Drive Access

1. Create a Google Cloud Project:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Google Drive API:
     ```
     APIs & Services > Library > Search for "Google Drive API" > Enable
     ```

2. Create OAuth 2.0 Credentials:
   - Go to `APIs & Services > Credentials`
   - Click `Create Credentials > OAuth client ID`
   - Select `Desktop Application`
   - Download the credentials JSON file

3. Generate Access Token:
   ```python
   from google_auth_oauthlib.flow import InstalledAppFlow
   import json

   # If modifying scopes, delete token.json.
   SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

   def generate_token():
       # Load client configuration from downloaded credentials
       flow = InstalledAppFlow.from_client_config(
           # Your downloaded credentials
           client_config={
               "installed": {
                   "client_id": "YOUR_CLIENT_ID",
                   "project_id": "YOUR_PROJECT_ID",
                   "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                   "token_uri": "https://oauth2.googleapis.com/token",
                   "client_secret": "YOUR_CLIENT_SECRET",
                   "redirect_uris": ["http://localhost"]
               }
           },
           scopes=SCOPES
       )

       # Run local server for authentication
       creds = flow.run_local_server(port=0)

       # Save the credentials
       token_data = {
           'token': creds.token,
           'refresh_token': creds.refresh_token,
           'token_uri': creds.token_uri,
           'client_id': creds.client_id,
           'client_secret': flow.client_config['installed']['client_secret'],
           'scopes': creds.scopes
       }

       with open('token.json', 'w') as token_file:
           json.dump(token_data, token_file)

   if __name__ == '__main__':
       generate_token()
   ```

4. Set up environment variables:
   ```env
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/token.json
   ```

### Using Google Drive Source

```python
from vector_search import VectorSearch

# Initialize with Google Drive source
vector_search = VectorSearch(
    source_type="google_drive",
    chunker_type="word",
    embedding_type="ollama",
    database_type="postgres"
)

# Process documents from a Google Drive folder
folder_id = "your_folder_id"  # Get this from the folder's URL
vector_search.process_source(folder_id)
```

### Supported File Types

The Google Drive integration supports various file types:

1. Google Workspace Files (automatically converted to Markdown):
   - Google Docs (`application/vnd.google-apps.document`)
   - Google Sheets (`application/vnd.google-apps.spreadsheet`)
   - Google Slides (`application/vnd.google-apps.presentation`)

2. Regular Files:
   - Text files (`text/plain`)
   - Markdown files (`text/markdown`)
   - JSON files (`application/json`)
   - PDF files (`application/pdf`)
   - HTML files (`text/html`)

### Finding Folder ID

To get a folder's ID:
1. Open the folder in Google Drive
2. The URL will look like: `https://drive.google.com/drive/folders/1234...`
3. The long string after `folders/` is your folder ID

### Example Usage

```python
from vector_search.sources import GoogleDriveSource

# Initialize the source
source = GoogleDriveSource()

# Process files from a folder
for doc in source.load_documents("your_folder_id"):
    print(f"File: {doc['metadata']['source']}")
    print(f"Format: {doc['metadata']['format']}")
    print(f"Original MIME type: {doc['metadata']['original_mime_type']}")
    print(f"Content preview: {doc['text'][:100]}...")
```

### Metadata

Each document includes metadata:
```python
{
    "source": "filename_without_extension",
    "format": "md",  # Format after conversion (e.g., 'md' for Google Docs)
    "drive_id": "google_drive_file_id",
    "mime_type": "text/markdown",  # Export format
    "original_mime_type": "application/vnd.google-apps.document"  # Original format
}
```

### Error Handling

The system handles errors gracefully:
- Invalid files are skipped with error messages
- Unsupported formats are ignored
- Authentication errors are reported clearly

If you encounter authentication errors:
1. Delete the `token.json` file
2. Run the token generation script again
3. Make sure your Google account has access to the folder
