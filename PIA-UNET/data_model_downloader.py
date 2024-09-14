import gdown
import zipfile
import os

# Step 1: Define the Google Drive file ID and output zip file path
file_id = '1OxKZDAfPPXU4zfeoK5-hnTDJuMZKm0XA'
download_url = f'https://drive.google.com/uc?id={file_id}'
output_zip = 'downloaded_file.zip'

# Step 2: Download the zip file from Google Drive
gdown.download(download_url, output_zip, quiet=False)

# Step 3: Unzip the file in the current directory
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall()
    print(f"File unzipped into the current directory.")

# Step 4: Remove the zip file after extraction
os.remove(output_zip)
print(f"Zip file '{output_zip}' has been removed.")

