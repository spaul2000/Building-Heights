import rasterio
from google.cloud import storage
import io
import os
import numpy as np

# TO START: run 'gcloud auth application-default login' and login with your personal email

def read_and_preprocess_n_geotiffs_from_gcs(bucket_name, folder_path, n, bands_to_keep):
    """Read and preprocess the first 'n' GeoTIFFs from a folder in GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blobs = list(bucket.list_blobs(prefix=folder_path))[:n]
    
    processed_data = []
    profiles = []  # Store profiles (metadata) for each cropped segment
    
    for blob in blobs:
        # Read the GeoTIFF from GCS into memory
        in_memory_file = io.BytesIO(blob.download_as_bytes())
        
        with rasterio.open(in_memory_file) as src:
            print(f"Height of GeoTIFF: {src.height}, Width of GeoTIFF: {src.width}")
            # Extract and store the original profile
            original_profile = src.profile
            
            # Extract desired bands
            selected_bands = src.read(bands_to_keep)
            
            # Append data and profiles to their respective lists
            for y in range(0, selected_bands.shape[1], 128):
                for x in range(0, selected_bands.shape[2], 128):
                    cropped_data = selected_bands[:, y:y+128, x:x+128]
                    
                    if cropped_data.shape[1] == 128 and cropped_data.shape[2] == 128:
                            processed_data.append(cropped_data)
                            cropped_profile = src.profile.copy()
                            cropped_profile["height"] = 128
                            cropped_profile["width"] = 128
                            cropped_profile["count"] = len(bands_to_keep)
                            cropped_profile["transform"] = rasterio.windows.transform(
                                window=rasterio.windows.Window(x, y, 128, 128),
                                transform=src.transform
                            )
                    profiles.append(cropped_profile)
    
    return processed_data, profiles

def save_cropped_tiffs(data_list, profiles, output_folder, prefix="cropped"):
    """Save each data array in the list as a GeoTIFF file using the provided profiles."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    saved_files = []
    
    for i, (data, profile) in enumerate(zip(data_list, profiles)):
        output_path = os.path.join(output_folder, f"{prefix}_{i}.tif")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data.astype(np.float32))
        saved_files.append(output_path)
        
    return saved_files


bucket_name = 'cs325b-building-height'
folder_path = 'data/sat_img/tx_sample_gt_2000/'
bands_to_keep = [1]  
n = 1  # Number of geotiffs to read and preprocess

# Call the reading and preprocessing function
processed_data, profiles = read_and_preprocess_n_geotiffs_from_gcs(bucket_name, folder_path, n, bands_to_keep)

# Save the processed data to individual GeoTIFF files in a local folder
output_folder = "./data/mean_crops"
saved_files = save_cropped_tiffs(processed_data, profiles, output_folder)

# print the paths of the saved files
for file_path in saved_files:
    print(file_path)
