"""Input source handlers for vector search."""
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Set, Union, Callable, Optional

from azure.storage.blob import BlobServiceClient
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


class BaseSource(ABC):
    """Base class for input sources."""

    def __init__(self, supported_formats: Set[str] = None, text_filter: Optional[Callable[[str], str]] = None):
        """Initialize input source.

        Args:
            supported_formats: Set of supported file formats (e.g., {'txt', 'pdf'})
            text_filter: Optional function that takes a string and returns a filtered string
        """
        self.supported_formats = supported_formats or {'txt', 'md', 'markdown', 'json', 'pdf'}
        self.text_filter = text_filter

    @abstractmethod
    def load_documents(self, source_path: str) -> Iterator[Dict]:
        """Load documents from the source.

        Args:
            source_path: Path or identifier for the source

        Yields:
            Dictionary containing document text and metadata
        """
        pass

    def _is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported.

        Args:
            file_path: Path to the file

        Returns:
            True if format is supported, False otherwise
        """
        return Path(file_path).suffix.lower()[1:] in self.supported_formats

    def _read_file(self, file_path: Union[str, Path]) -> Dict:
        """Read file content based on its format.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file content and metadata
        """
        file_path = Path(file_path)
        file_format = file_path.suffix.lower()[1:]

        if not self._is_supported_format(file_path):
            raise ValueError(f"Unsupported file format: {file_format}")

        content = ""
        if file_format == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif file_format == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.dumps(json.load(f), indent=2)
        elif file_format == "md" or file_format == "markdown":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif file_format == "pdf":
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            content = " ".join(page.get_text() for page in doc)

        # Apply text filter if provided
        if self.text_filter is not None:
            content = self.text_filter(content)

        return {
            "text": content,
            "metadata": {
                "source": file_path.stem,
                "format": file_format,
                "path": str(file_path)
            }
        }


class FolderSource(BaseSource):
    """Handler for loading documents from a folder."""

    def __init__(self, supported_formats: Set[str] = None, text_filter: Optional[Callable[[str], str]] = None):
        """Initialize folder source.

        Args:
            supported_formats: Set of supported file formats
            text_filter: Optional function that takes a string and returns a filtered string
        """
        super().__init__(supported_formats, text_filter)

    def load_documents(self, source_path: str) -> Iterator[Dict]:
        """Load documents from a folder recursively.

        Args:
            source_path: Path to the folder

        Yields:
            Dictionary containing document text and metadata
        """
        folder_path = Path(source_path)
        if not folder_path.is_dir():
            raise ValueError(f"Not a directory: {source_path}")

        for file_path in folder_path.rglob("*"):
            if file_path.is_file() and self._is_supported_format(file_path):
                yield self._read_file(file_path)


class FileSource(BaseSource):
    """Handler for loading a single document."""

    def __init__(self, supported_formats: Set[str] = None, text_filter: Optional[Callable[[str], str]] = None):
        """Initialize file source.

        Args:
            supported_formats: Set of supported file formats
            text_filter: Optional function that takes a string and returns a filtered string
        """
        super().__init__(supported_formats, text_filter)

    def load_documents(self, source_path: str) -> Iterator[Dict]:
        """Load a single document.

        Args:
            source_path: Path to the file

        Yields:
            Dictionary containing document text and metadata
        """
        file_path = Path(source_path)
        if not file_path.is_file():
            raise ValueError(f"Not a file: {source_path}")

        if self._is_supported_format(file_path):
            yield self._read_file(file_path)


class GoogleDriveSource(BaseSource):
    """Handler for loading documents from Google Drive."""

    # Google Workspace MIME types and their export formats
    GOOGLE_MIME_TYPES = {
        'application/vnd.google-apps.document': 'text/markdown',  # Google Docs to Markdown
        'application/vnd.google-apps.spreadsheet': 'text/markdown',  # Sheets to Markdown
        'application/vnd.google-apps.presentation': 'text/markdown',  # Slides to Markdown
    }

    # Common MIME type mappings
    MIME_TYPE_FORMATS = {
        'text/plain': 'txt',
        'text/markdown': 'md',
        'application/json': 'json',
        'application/pdf': 'pdf',
        'text/markdown': 'md',
        'text/html': 'html',
    }

    def __init__(self, supported_formats: Set[str] = None, text_filter: Optional[Callable[[str], str]] = None):
        """Initialize Google Drive source.

        Args:
            supported_formats: Set of supported file formats
            text_filter: Optional function that takes a string and returns a filtered string
        """
        super().__init__(supported_formats, text_filter)
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError("Google Drive credentials path not set")

        credentials = Credentials.from_authorized_user_file(
            credentials_path,
            ["https://www.googleapis.com/auth/drive.readonly"]
        )
        self.service = build("drive", "v3", credentials=credentials)

    def _is_google_workspace_file(self, mime_type: str) -> bool:
        """Check if the file is a Google Workspace document.

        Args:
            mime_type: MIME type of the file

        Returns:
            True if it's a Google Workspace document, False otherwise
        """
        return mime_type in self.GOOGLE_MIME_TYPES

    def _get_format_from_mime_type(self, mime_type: str) -> str:
        """Get format from MIME type.

        Args:
            mime_type: MIME type of the file

        Returns:
            Format string (e.g., 'md', 'txt', 'pdf')
        """
        return self.MIME_TYPE_FORMATS.get(mime_type, 'txt')  # Default to txt if unknown

    def _process_file_content(self, file_bytes: bytes, mime_type: str) -> str:
        """Process file content based on its MIME type.

        Args:
            file_bytes: Raw file content
            mime_type: MIME type of the file

        Returns:
            Extracted text content
        """
        if mime_type == 'application/pdf':
            import fitz  # PyMuPDF
            import io
            
            # Create a PDF document from the bytes
            with io.BytesIO(file_bytes) as pdf_stream:
                doc = fitz.open(stream=pdf_stream, filetype="pdf")
                return " ".join(page.get_text() for page in doc)
                
        elif mime_type in ['text/plain', 'text/markdown', 'text/html']:
            return file_bytes.decode("utf-8")
            
        elif mime_type == 'application/json':
            content = json.loads(file_bytes.decode("utf-8"))
            return json.dumps(content, indent=2)
            
        else:
            return file_bytes.decode("utf-8")

    def _download_google_workspace_file(self, file_id: str, mime_type: str) -> tuple[bytes, str]:
        """Download and export Google Workspace file.

        Args:
            file_id: ID of the Google Workspace file
            mime_type: MIME type of the file

        Returns:
            Tuple of (file content as bytes, export mime type)
        """
        import io

        export_mime_type = self.GOOGLE_MIME_TYPES[mime_type]
        request = self.service.files().export_media(fileId=file_id, mimeType=export_mime_type)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            _, done = downloader.next_chunk()

        return fh.getvalue(), export_mime_type

    def _download_regular_file(self, file_id: str) -> bytes:
        """Download regular (non-Google Workspace) file.

        Args:
            file_id: ID of the file

        Returns:
            File content as bytes
        """
        import io

        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            _, done = downloader.next_chunk()

        return fh.getvalue()

    def load_documents(self, source_path: str) -> Iterator[Dict]:
        """Load documents from Google Drive folder.

        Args:
            source_path: Google Drive folder ID

        Yields:
            Dictionary containing document text and metadata
        """
        query = f"'{source_path}' in parents"
        results = self.service.files().list(
            q=query,
            fields="files(id, name, mimeType)"
        ).execute()

        print(f"Found files: {[f['name'] for f in results.get('files', [])]}")

        for file in results.get("files", []):
            try:
                mime_type = file["mimeType"]
                file_name = file["name"]
                file_id = file["id"]

                print(f"Processing {file_name} (MIME type: {mime_type})")

                # Handle Google Workspace documents
                if self._is_google_workspace_file(mime_type):
                    # Get the exported content as markdown
                    file_bytes, export_mime_type = self._download_google_workspace_file(file_id, mime_type)
                    content = self._process_file_content(file_bytes, export_mime_type)
                    format_type = self._get_format_from_mime_type(export_mime_type)
                
                # Handle regular files
                else:
                    file_bytes = self._download_regular_file(file_id)
                    content = self._process_file_content(file_bytes, mime_type)
                    format_type = self._get_format_from_mime_type(mime_type)

                # Apply text filter if provided
                if self.text_filter is not None:
                    content = self.text_filter(content)

                yield {
                    "text": content,
                    "metadata": {
                        "source": Path(file_name).stem,
                        "format": format_type,
                        "drive_id": file_id,
                        "mime_type": mime_type,
                        "original_mime_type": mime_type
                    }
                }
            except Exception as e:
                print(f"Error processing file {file.get('name', 'unknown')}: {str(e)}")
                continue


class AzureBlobSource(BaseSource):
    """Handler for loading documents from Azure Blob Storage."""

    def __init__(self, supported_formats: Set[str] = None, text_filter: Optional[Callable[[str], str]] = None):
        """Initialize Azure Blob source.

        Args:
            supported_formats: Set of supported file formats
            text_filter: Optional function that takes a string and returns a filtered string
        """
        super().__init__(supported_formats, text_filter)
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER")

        if not connection_string or not self.container_name:
            raise ValueError("Azure Blob connection string and container name must be set")

        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

    def load_documents(self, source_path: str) -> Iterator[Dict]:
        """Load documents from Azure Blob container.

        Args:
            source_path: Prefix path in the container

        Yields:
            Dictionary containing document text and metadata
        """
        blobs = self.container_client.list_blobs(name_starts_with=source_path)
        
        for blob in blobs:
            if self._is_supported_format(blob.name):
                blob_client = self.container_client.get_blob_client(blob.name)
                content = blob_client.download_blob().readall().decode("utf-8")
                
                yield {
                    "text": content,
                    "metadata": {
                        "source": Path(blob.name).stem,
                        "format": Path(blob.name).suffix.lower()[1:],
                        "container": self.container_name,
                        "blob_path": blob.name
                    }
                } 