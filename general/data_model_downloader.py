import gdown
import zipfile
import os

# Step 1: Define the Google Drive file ID and output zip file path
file_id = '1bpYfksqKwGzaxm4htFtQr0iCaWfQBRKZ'
download_url = f'https://drive.google.com/uc?id={file_id}'
output_zip = 'downloaded_file.zip'

# Step 2: Download the file from Google Drive
gdown.download(download_url, output_zip, quiet=False)

# Step 3: Unzip the downloaded file
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall()  # Extract to the current directory (you can change the destination)
    print("File unzipped successfully.")

# Step 4: Remove the zip file
os.remove(output_zip)
print(f"Zip file '{output_zip}' has been removed.")
