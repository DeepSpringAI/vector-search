"""Test input source providers."""
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Iterator

from vector_search.sources import (
    FolderSource, FileSource, GoogleDriveSource, AzureBlobSource
)


def create_test_file(path: Path, content: str) -> None:
    """Create a test file with given content."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_folder_source():
    """Test folder source provider."""
    source = FolderSource()
    
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
        create_test_file(test_folder / filename, content)
    
    print("\nFolder Source Results:")
    for doc in source.load_documents(str(test_folder)):
        print(doc)
        input('Press Enter to continue...')
        # print(f"\nFile: {doc['metadata']['source']}")
        # print(f"Format: {doc['metadata']['format']}")
        # print(f"Text preview: {doc['text'][:50]}...")
    
    # Cleanup
    for file in test_folder.glob("*"):
        file.unlink()
    test_folder.rmdir()


def test_file_source():
    """Test single file source provider."""
    source = FileSource()
    
    # Create test file
    test_file = Path("test.txt")
    content = """
    This is a test file for the FileSource provider.
    It contains multiple lines of text.
    We'll use this to test single file loading.
    """
    
    create_test_file(test_file, content)
    
    print("\nFile Source Results:")
    for doc in source.load_documents(str(test_file)):
        print(f"\nFile: {doc['metadata']['source']}")
        print(f"Format: {doc['metadata']['format']}")
        print(f"Text preview: {doc['text'][:50]}...")
    
    # Cleanup
    test_file.unlink()


def test_google_drive_source():
    """Test Google Drive source provider."""
    load_dotenv()
    
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("\nSkipping Google Drive test - credentials not set")
        return
        
    source = GoogleDriveSource()
    folder_id = "your_folder_id"  # Replace with actual folder ID
    
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
    load_dotenv()
    
    if not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        print("\nSkipping Azure Blob test - connection string not set")
        return
        
    source = AzureBlobSource()
    blob_prefix = "test/"  # Replace with actual blob prefix
    
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


def test_text_filter_functionality():
    """Test the text filter functionality across different sources."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_content = "Hello World! This is a TEST."
        test_file = Path(temp_dir) / "test.txt"
        create_test_file(test_file, test_content)

        # Test with lowercase filter
        def lowercase_filter(text: str) -> str:
            return text.lower()

        file_source = FileSource(text_filter=lowercase_filter)
        docs = list(file_source.load_documents(str(test_file)))
        print("\nTesting lowercase filter:")
        print(f"Original: {test_content}")
        print(f"Filtered: {docs[0]['text']}")
        
        # Test with punctuation removal filter
        def remove_punctuation_filter(text: str) -> str:
            return "".join(char for char in text if char.isalnum() or char.isspace())

        file_source = FileSource(text_filter=remove_punctuation_filter)
        docs = list(file_source.load_documents(str(test_file)))
        print("\nTesting punctuation removal filter:")
        print(f"Original: {test_content}")
        print(f"Filtered: {docs[0]['text']}")


def test_multiple_files_with_filter():
    """Test text filter with multiple files in a folder."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple test files
        files = {
            "test1.txt": "HELLO WORLD",
            "test2.txt": "TESTING 123",
            "test3.md": "# MARKDOWN TEST"
        }
        
        for filename, content in files.items():
            create_test_file(Path(temp_dir) / filename, content)

        def lowercase_filter(text: str) -> str:
            return text.lower()

        folder_source = FolderSource(text_filter=lowercase_filter)
        docs = list(folder_source.load_documents(temp_dir))
        
        print("\nTesting multiple files with lowercase filter:")
        for doc in docs:
            print(f"\nFile: {doc['metadata']['source']}")
            print(f"Original: {files[doc['metadata']['source'] + '.' + doc['metadata']['format']]}")
            print(f"Filtered: {doc['text']}")


def main():
    """Run all tests."""
    print("Testing Folder Source:")
    test_folder_source()
    
    # print("\nTesting File Source:")
    # test_file_source()
    
    # print("\nTesting Google Drive Source:")
    # test_google_drive_source()
    
    # print("\nTesting Azure Blob Source:")
    # test_azure_blob_source()
    
    # print("\nTesting Text Filter Functionality:")
    # test_text_filter_functionality()
    
    # print("\nTesting Multiple Files with Filter:")
    # test_multiple_files_with_filter()


if __name__ == "__main__":
    main() 