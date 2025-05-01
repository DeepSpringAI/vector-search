# Vector Search

<div align="center">
  <img src="inc/logo.png" alt="Vector Search Logo" width="200"/>
</div>

A flexible and modular vector search system for document processing, embedding generation, and similarity search.

## Key Features

### Input Sources
- 📁 **Multiple Source Providers**
  - Local folder and file processing
  - Google Drive integration with Google Workspace support
  - Azure Blob Storage integration
  - Custom source provider support

### Text Processing
- 📝 **Flexible Text Chunking**
  - Word-based chunking
  - Character-based chunking
  - Custom chunking strategies
  - Configurable chunk size and overlap

- 🔄 **Text Augmentation**
  - Multiple AI providers (Ollama, OpenAI, Azure OpenAI)
  - Semantic variation generation
  - Original content preservation
  - Customizable augmentation parameters

- 🏷️ **AI-Powered Tag Generation**
  - Free-form tag generation
  - Predefined tag selection
  - Multiple model support (Ollama, OpenAI, Azure)
  - Customizable tag parameters

### Embeddings
- 🧠 **Multiple Embedding Providers**
  - Ollama (local processing)
  - OpenAI
  - Azure OpenAI
  - Custom embedding support

### Storage & Search
- 💾 **Database Options**
  - PostgreSQL with pgvector
  - Supabase integration
  - Vector similarity search
  - Metadata storage and filtering

### Additional Features
- 🔍 **Text Filtering**
  - Custom preprocessing filters
  - Format-specific handling
  - Metadata preservation

- 🔐 **Security**
  - Environment-based configuration
  - API key management
  - Secure credential handling

- 🛠️ **Extensibility**
  - Custom component support
  - Modular architecture
  - Easy integration options

## Table of Contents

1. [Installation](#installation)
2. [Environment Setup](#environment-setup)
3. [Basic Usage](#basic-usage)
4. [Components](#components)
   - [Source Providers](#source-providers)
   - [Text Processing](#text-processing)
   - [Embedding Providers](#embedding-providers)
   - [Database Integration](#database-integration)
5. [Advanced Features](#advanced-features)
   - [Text Augmentation](#text-augmentation)
   - [Tag Generation](#tag-generation)
   - [Text Filtering](#text-filtering)
6. [Custom Components](#custom-components)
7. [Testing](#testing)
8. [Contributing](#contributing)

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

# AI Provider Configuration
OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
OLLAMA_BASE_URL=http://localhost:11434

# Storage Provider Configuration
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_STORAGE_CONTAINER=your_container_name
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

## Components

### Source Providers

#### Local Sources
```python
from vector_search import VectorSearch

# Using folder source
vs = VectorSearch(source_type="folder")
vs.process_source("path/to/folder")

# Using single file source
vs = VectorSearch(source_type="file")
vs.process_source("path/to/file.txt")
```

#### Google Drive Integration
```python
from vector_search import VectorSearch

vs = VectorSearch(source_type="google_drive")
vs.process_source("your_folder_id")  # From folder URL
```

Supports:
- Google Docs (→ Markdown)
- Google Sheets (→ Markdown)
- Google Slides (→ Markdown)
- PDFs, Text files, etc.

#### Azure Blob Storage
```python
from vector_search import VectorSearch

vs = VectorSearch(source_type="azure_blob")
vs.process_source("container/path")
```

### Text Processing

#### Chunking Strategies
```python
from vector_search import VectorSearch

# Word-based chunking
vs = VectorSearch(
    chunker_type="word",
    chunk_size=1000,
    chunk_overlap=200
)

# Character-based chunking
vs = VectorSearch(
    chunker_type="character",
    chunk_size=4000,
    chunk_overlap=400
)
```

#### Text Augmentation
```python
from vector_search import VectorSearch

# Using Ollama for augmentation
vs = VectorSearch(
    augment=True,
    augmenter_type="ollama",
    augmenter_config={
        "model_name": "llama3.1:8b"
    }
)

# Using OpenAI
vs = VectorSearch(
    augment=True,
    augmenter_type="openai",
    augmenter_config={
        "model_name": "gpt-3.5-turbo"
    }
)
```

#### Tag Generation
```python
from vector_search.tags import OllamaTagGenerator, OpenAITagGenerator

# Free-form tags
generator = OllamaTagGenerator(max_tags=3)
tagged_chunks = generator.generate_tags(chunks)

# Predefined tags
generator = OpenAITagGenerator(
    max_tags=3,
    predefined_tags={"ai", "machine learning", "python"}
)
tagged_chunks = generator.generate_tags(chunks)
```

### Embedding Providers

#### Ollama (Local)
```python
from vector_search import VectorSearch

vs = VectorSearch(
    embedding_type="ollama",
    embedding_config={
        "model_name": "bge-m3:latest"
    }
)
```

#### OpenAI
```python
vs = VectorSearch(
    embedding_type="openai",
    embedding_config={
        "model_name": "text-embedding-3-small"
    }
)
```

#### Azure OpenAI
```python
vs = VectorSearch(
    embedding_type="azure_openai",
    embedding_config={
        "deployment": "your-deployment"
    }
)
```

### Database Integration

#### PostgreSQL
```python
from vector_search import VectorSearch

vs = VectorSearch(database_type="postgres")
```

Required schema:
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

#### Supabase
```python
vs = VectorSearch(database_type="supabase")
```

Additional function for Supabase:
```sql
create or replace function match_chunks (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  text text,
  metadata jsonb,
  date timestamptz,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    chunks.id,
    chunks.text,
    chunks.metadata,
    chunks.date,
    1 - (chunks.embedding <=> query_embedding) as similarity
  from chunks
  where 1 - (chunks.embedding <=> query_embedding) > match_threshold
  order by chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

## Advanced Features

### Text Filtering
```python
def custom_filter(text: str) -> str:
    # Remove specific patterns
    text = text.replace("[REMOVE]", "")
    # Convert to lowercase
    text = text.lower()
    # Clean whitespace
    text = " ".join(text.split())
    return text

vs = VectorSearch(
    source_type="folder",
    text_filter=custom_filter
)
```

### Custom Components

#### Custom Source
```python
from vector_search.sources import BaseSource

class CustomSource(BaseSource):
    def load_documents(self, source_path: str):
        # Your implementation
        yield {
            "text": content,
            "metadata": metadata
        }
```

#### Custom Chunker
```python
from vector_search.chunker import BaseChunker

class CustomChunker(BaseChunker):
    def chunk_text(self, text: str, metadata: dict = None):
        # Your implementation
        return chunks
```

#### Custom Embedding
```python
from vector_search.embeddings import BaseEmbedding

class CustomEmbedding(BaseEmbedding):
    def embed(self, texts: Union[str, List[str]]):
        # Your implementation
        return embeddings
```

## Testing

Run specific tests:
```bash
python -m pytest tests/test_database.py
python -m pytest tests/test_embeddings.py
python -m pytest tests/test_sources.py
```

Run all tests:
```bash
python -m pytest tests/
```

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

# Text Augmentation

The vector-search library supports text augmentation to enhance search quality by generating semantically similar variations of your text chunks. This can help improve recall and make the search more robust.

## Available Augmenters

### 1. Ollama Augmenter
Uses Ollama's local LLMs to generate text variations. Ideal for when you want to keep data processing local or have privacy requirements.

```python
from vector_search.augmentation import OllamaAugmenter

augmenter = OllamaAugmenter(
    model_name="llama3.1:8b",  # Default model
    base_url="http://localhost:11434"  # Optional, defaults to OLLAMA_BASE_URL env var
)
```

### 2. OpenAI Augmenter
Uses OpenAI's GPT models to generate high-quality text variations.

```python
from vector_search.augmentation import OpenAIAugmenter

augmenter = OpenAIAugmenter(
    api_key="your_api_key",  # Optional, defaults to OPENAI_API_KEY env var
    model_name="gpt-3.5-turbo"  # Default model
)
```

### 3. Azure OpenAI Augmenter
Uses Azure's OpenAI service for text augmentation, suitable for enterprise deployments.

```python
from vector_search.augmentation import AzureOpenAIAugmenter

augmenter = AzureOpenAIAugmenter(
    api_key=None,  # Optional, defaults to AZURE_OPENAI_API_KEY env var
    api_version=None,  # Optional, defaults to AZURE_OPENAI_API_VERSION env var
    endpoint=None,  # Optional, defaults to AZURE_OPENAI_ENDPOINT env var
    deployment="text-davinci-003"  # Default deployment
)
```

## Usage Example

```python
# Prepare your text chunks
chunks = [
    {
        "text": "Machine learning enables systems to learn from experience.",
        "metadata": {
            "source": "doc.txt"},
        "chunk_index": 0
    }
]

# Initialize an augmenter
augmenter = OllamaAugmenter()

# Generate augmented chunks
augmented_chunks = augmenter.augment(chunks)

# Each original chunk will be preserved and augmented versions will be added
for chunk in augmented_chunks:
    print(f"Text: {chunk['text']}")
    if chunk['metadata'].get('augmented'):
        print(f"Augmented version of chunk {chunk['metadata']['original_chunk_index']}")
```

## Environment Variables

Configure your augmenters using these environment variables in your `.env` file:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_ENDPOINT=your_endpoint
```

## How It Works

1. Each augmenter preserves the original chunks and adds augmented versions
2. Augmented chunks include metadata indicating:
   - `augmented: true` - Marks it as an augmented version
   - `original_chunk_index` - References the source chunk
3. The augmentation process maintains the core meaning while varying the text structure and wording
4. All metadata from the original chunk is preserved in the augmented versions

## Best Practices

1. Choose the appropriate augmenter based on your needs:
   - Ollama for local processing and privacy
   - OpenAI for high-quality results
   - Azure OpenAI for enterprise deployments
2. Test augmented results to ensure they maintain semantic accuracy
3. Consider the trade-off between augmentation quantity and processing time
4. Use environment variables for API keys and configuration

# Tag Generation

The vector-search library includes powerful AI-powered tag generation capabilities. You can either generate free-form tags or select from a predefined list of tags. The feature supports multiple AI providers:

## Tag Generators

### 1. Free-form Tag Generation

Generate tags freely based on the content:

#### Using Ollama (Local)
```python
from vector_search.tags import OllamaTagGenerator

generator = OllamaTagGenerator(
    model_name="llama3.1:8b",  # Model to use
    max_tags=3,                # Maximum tags per chunk
    temperature=0.3,           # Model creativity (0.0 to 1.0)
    base_url=None             # Optional: Defaults to OLLAMA_BASE_URL env var
)

# Process chunks
chunks = [
    {
        "text": "Machine learning enables systems to learn from experience.",
        "metadata": {"source": "doc.txt"},
        "chunk_index": 0
    }
]

tagged_chunks = generator.generate_tags(chunks)
# Result: {"tags": ["machine learning", "artificial intelligence", "automation"]}
```

#### Using OpenAI
```python
from vector_search.tags import OpenAITagGenerator

generator = OpenAITagGenerator(
    model_name="gpt-3.5-turbo",  # OpenAI model to use
    max_tags=3,
    temperature=0.3,
    api_key=None               # Optional: Defaults to OPENAI_API_KEY env var
)
```

#### Using Azure OpenAI
```python
from vector_search.tags import AzureOpenAITagGenerator

generator = AzureOpenAITagGenerator(
    deployment="your-deployment",
    max_tags=3,
    temperature=0.3,
    api_key=None,             # Optional: From AZURE_OPENAI_API_KEY
    api_version=None,         # Optional: From AZURE_OPENAI_API_VERSION
    endpoint=None             # Optional: From AZURE_OPENAI_ENDPOINT
)
```

### 2. Predefined Tag Selection

Select tags from a predefined list:

#### Using Ollama
```python
from vector_search.tags import OllamaPredefinedTagSelector

# Define allowed tags
predefined_tags = {
    "machine learning",
    "artificial intelligence",
    "programming",
    "python",
    "data science",
    "deep learning"
}

selector = OllamaPredefinedTagSelector(
    predefined_tags=predefined_tags,
    model_name="llama3.1:8b",
    max_tags=3
)

# Process chunks
tagged_chunks = selector.generate_tags(chunks)
# Result: {"tags": ["machine learning", "artificial intelligence"]}
```

#### Using OpenAI
```python
from vector_search.tags import OpenAIPredefinedTagSelector

selector = OpenAIPredefinedTagSelector(
    predefined_tags=predefined_tags,
    model_name="gpt-3.5-turbo",
    max_tags=3
)
```

#### Using Azure OpenAI
```python
from vector_search.tags import AzureOpenAIPredefinedTagSelector

selector = AzureOpenAIPredefinedTagSelector(
    predefined_tags=predefined_tags,
    deployment="your-deployment",
    max_tags=3
)
```

## Configuration

### Environment Variables

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_ENDPOINT=your_endpoint
```

## Features

1. **Multiple AI Providers**:
   - Ollama for local processing
   - OpenAI for cloud processing
   - Azure OpenAI for enterprise deployments

2. **Two Tag Generation Modes**:
   - Free-form tag generation
   - Selection from predefined tags

3. **Customization Options**:
   - Control number of tags per chunk
   - Adjust model temperature
   - Choose specific models/deployments

4. **Error Handling**:
   - Graceful handling of API errors
   - Preservation of original chunks on failure
   - Validation of predefined tags

5. **Metadata Integration**:
   - Tags added to chunk metadata
   - Original metadata preserved
   - Chunk indices maintained

## Output Format

Each processed chunk includes:
```python
{
    "text": "Original text content",
    "metadata": {
        "source": "original_source",
        "format": "txt",
        "tags": ["tag1", "tag2", "tag3"],  # Generated/selected tags
        # ... other original metadata ...
    },
    "chunk_index": 0
}
```

## Best Practices

1. **Model Selection**:
   - Use Ollama for privacy-sensitive data
   - Use OpenAI for high-quality results
   - Use Azure OpenAI for enterprise compliance

2. **Tag Generation**:
   - Keep max_tags reasonable (2-5 recommended)
   - Use lower temperature (0.1-0.3) for consistency
   - Use higher temperature (0.5-0.7) for creativity

3. **Predefined Tags**:
   - Keep tag list focused and relevant
   - Use consistent formatting
   - Consider hierarchical relationships

4. **Error Handling**:
   - Always check for empty tag lists
   - Validate predefined tags
   - Handle API rate limits

## Dependencies

Required packages:
```bash
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.15
```
