from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
import json
import os

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def main():
    """Generate token for Google Drive API access."""
    # Load client configuration
    client_config = {
        "installed": {
            "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
            "project_id": "your-project-id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "YOUR_CLIENT_SECRET",
            "redirect_uris": ["http://localhost"]
        }
    }

    # Create flow instance
    flow = InstalledAppFlow.from_client_config(
        client_config,
        SCOPES
    )

    # Run the OAuth flow
    creds = flow.run_local_server(port=0)

    # Save the credentials
    token_data = {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': client_config['installed']['client_secret'],
        'scopes': creds.scopes
    }

    with open('token.json', 'w') as token_file:
        json.dump(token_data, token_file)
    
    print("Token has been generated and saved to 'token.json'")
    print("Please set GOOGLE_APPLICATION_CREDENTIALS to point to this token.json file")

if __name__ == '__main__':
    main() 