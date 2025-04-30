from vector_search.sources import GoogleDriveSource
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Google Drive source
    source = GoogleDriveSource()
    
    # Replace this with your Google Drive folder ID
    folder_id = "13IDuCnHVO9Ral_lbCSUnOLg0C94Ti680"  # Get this from your Google Drive folder URL
    
    print("Testing Google Drive access...")
    try:
        for doc in source.load_documents(folder_id):
            print("\nFound document:")
            print(f"Name: {doc['metadata']['source']}")
            print(f"Format: {doc['metadata']['format']}")
            print(f"Drive ID: {doc['metadata']['drive_id']}")
            print(f"Text preview: {doc['text'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 