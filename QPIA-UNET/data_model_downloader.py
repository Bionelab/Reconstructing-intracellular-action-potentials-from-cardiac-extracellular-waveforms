import gdown
import os
import zipfile
#### RUN THIS CODE IN TERMINAL
# Step 1: Function to download and save files from Google Drive
def download_file_from_google_drive(file_id, destination):
    drive_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(drive_url, destination, quiet=False)
    print(f"File saved at {destination}")

# Step 2: Create 'saved_models' folder if it doesn't exist and download the .h5 file
saved_models_dir = 'saved_models'
os.makedirs(saved_models_dir, exist_ok=True)
h5_file_path = os.path.join(saved_models_dir, 'data-4__seed-20__kernel_size-11__ch_num-32__depth-8__wph-0.02__lr-0.005__epoch-150__physics-True__max_sample-300__.h5')
h5_file_id = '1SEo0TK8RRFR-dIJBz2X3U4QTxhDlqjeh'
download_file_from_google_drive(h5_file_id, h5_file_path)

# Step 3: Create 'multichannels_data' folder if it doesn't exist and download the zip file
multi_data_dir = 'multichannels_data'
os.makedirs(multi_data_dir, exist_ok=True)
zip_file_path = os.path.join(multi_data_dir, 'multichannels_data.zip')
zip_file_id = '1sU1J15D-hZELCfbI8y2nDNX4vNNpWZva'
download_file_from_google_drive(zip_file_id, zip_file_path)

# Step 4: Unzip the downloaded file into the 'multichannels_data' folder and rename to multichannels_data.npy
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(multi_data_dir)
    print(f"File unzipped in {multi_data_dir}")

# Step 5: Rename the extracted file (assuming the extracted file is named 'extracted_file.npy')
# Ensure the name of the unzipped file is correct, and adjust this part if needed
extracted_file_name = 'multichannels_data.npy'  # Replace this with the actual name of the file in the zip
extracted_file_path = os.path.join(multi_data_dir, extracted_file_name)
# Step 6: Optionally, remove the zip file after extraction
os.remove(zip_file_path)
print(f"Zip file {zip_file_path} removed.")
