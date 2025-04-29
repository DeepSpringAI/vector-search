"""Test input source providers."""
import os
from pathlib import Path

from vector_search.config import Config
from vector_search.sources import (
    FolderSource, FileSource, GoogleDriveSource, AzureBlobSource
)


def test_folder_source():
    """Test folder source provider."""
    config = Config("config.yaml")
    source = FolderSource(config)
    
    # Create test folder with sample files
    test_folder = Path("test_data")
    test_folder.mkdir(exist_ok=True)
    
    # Create sample files
    files = {
        "test1.txt": "This is a test text file.",
        "test2.md": "# Test Markdown\nThis is a markdown file.",
        "test3.json": '{"key": "This is a JSON file"}',
    }
    
    for filename, content in files.items():
        with open(test_folder / filename, "w") as f:
            f.write(content)
    
    print("\nFolder Source Results:")
    for doc in source.load_documents(str(test_folder)):
        print(f"\nFile: {doc['metadata']['source']}")
        print(f"Format: {doc['metadata']['format']}")
        print(f"Text preview: {doc['text'][:50]}...")
    
    # Cleanup
    for file in test_folder.glob("*"):
        file.unlink()
    test_folder.rmdir()


def test_file_source():
    """Test single file source provider."""
    config = Config("config.yaml")
    source = FileSource(config)
    
    # Create test file
    test_file = Path("test.txt")
    content = """
    This is a test file for the FileSource provider.
    It contains multiple lines of text.
    We'll use this to test single file loading.
    """
    
    with open(test_file, "w") as f:
        f.write(content)
    
    print("\nFile Source Results:")
    for doc in source.load_documents(str(test_file)):
        print(f"\nFile: {doc['metadata']['source']}")
        print(f"Format: {doc['metadata']['format']}")
        print(f"Text preview: {doc['text'][:50]}...")
    
    # Cleanup
    test_file.unlink()


def test_google_drive_source():
    """Test Google Drive source provider."""
    if not os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH"):
        print("\nSkipping Google Drive test - credentials not set")
        return
        
    config = Config("config.yaml")
    source = GoogleDriveSource(config)
    
    # Replace with a real folder ID from your Google Drive
    folder_id = "your_folder_id"
    
    print("\nGoogle Drive Source Results:")
    try:
        for doc in source.load_documents(folder_id):
            print(f"\nFile: {doc['metadata']['source']}")
            print(f"Format: {doc['metadata']['format']}")
            print(f"Drive ID: {doc['metadata']['drive_id']}")
            print(f"Text preview: {doc['text'][:50]}...")
    except Exception as e:
        print(f"Error accessing Google Drive: {e}")


def test_azure_blob_source():
    """Test Azure Blob Storage source provider."""
    if not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        print("\nSkipping Azure Blob test - connection string not set")
        return
        
    config = Config("config.yaml")
    source = AzureBlobSource(config)
    
    # Replace with your blob prefix
    blob_prefix = "test/"
    
    print("\nAzure Blob Source Results:")
    try:
        for doc in source.load_documents(blob_prefix):
            print(f"\nFile: {doc['metadata']['source']}")
            print(f"Format: {doc['metadata']['format']}")
            print(f"Container: {doc['metadata']['container']}")
            print(f"Blob path: {doc['metadata']['blob_path']}")
            print(f"Text preview: {doc['text'][:50]}...")
    except Exception as e:
        print(f"Error accessing Azure Blob Storage: {e}")


if __name__ == "__main__":
    print("Testing Folder Source:")
    test_folder_source()
    
    print("\nTesting File Source:")
    test_file_source()
    
    # print("\nTesting Google Drive Source:")
    # test_google_drive_source()
    
    # print("\nTesting Azure Blob Source:")
    # test_azure_blob_source() 