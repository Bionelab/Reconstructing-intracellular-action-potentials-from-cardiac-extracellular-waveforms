import os
import numpy as np
import re

def extract_number(filename):
    # Extract the first number from the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def load_data(base_path):
    # Initialize lists for n arrays, p arrays, and names
    n_arrays = []
    p_arrays = []
    names = []

    # List all .npy files in the directory and sort them
    files = sorted([f for f in os.listdir(base_path) if f.endswith('.npy')])

    # Separate the filenames into n and p groups with corresponding indices
    n_files = [f for f in files if f.startswith('n')]
    p_files = [f for f in files if f.startswith('p')]

    # Ensure nx is at the same index as px
    n_files, p_files = zip(*sorted(zip(n_files, p_files), key=lambda x: extract_number(x[0])))

    # Load each file and append to the appropriate list
    for n_file, p_file in zip(n_files, p_files):
        n_data = np.load(os.path.join(base_path, n_file))
        p_data = np.load(os.path.join(base_path, p_file))
        n_arrays.append(n_data)
        p_arrays.append(p_data)
        names.append(n_file[:-4])  # Append the name without '.npy'

    return n_arrays, p_arrays, names


