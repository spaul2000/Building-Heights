import rasterio
from google.cloud import storage
import io
import os
import numpy as np
import csv
from tqdm import tqdm
import random
# TO START: run 'gcloud auth application-default login' and login with your personal email

def read_and_preprocess_n_geotiffs_from_gcs(bucket_name, folder_path, n, bands_to_keep, mask_band):
    """Read and preprocess the first 'n' GeoTIFFs from a folder in GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blobs = list(bucket.list_blobs(prefix=folder_path))
    
    processed_data = []
    mask_data = []
    profiles = []  # Store profiles (metadata) for each cropped segment
    
    for blob in tqdm(list(bucket.list_blobs(prefix=folder_path)), desc="Processing GeoTIFFs"):
        # Read the GeoTIFF from GCS into memory
        in_memory_file = io.BytesIO(blob.download_as_bytes())
        
        with rasterio.open(in_memory_file) as src:
            print(f"Height of GeoTIFF: {src.height}, Width of GeoTIFF: {src.width}")
            
            # Extract desired bands
            selected_bands = src.read(bands_to_keep)
            mask_band_data = src.read(mask_band)
            
            # Append data and profiles to their respective lists
            for y in range(0, selected_bands.shape[1], 128):
                for x in range(0, selected_bands.shape[2], 128):
                    cropped_data = selected_bands[:, y:y+128, x:x+128]
                    cropped_mask = mask_band_data[:, y:y+128, x:x+128]
                    
                    if cropped_data.shape[1] == 128 and cropped_data.shape[2] == 128:
                        processed_data.append(cropped_data)
                        mask_data.append(cropped_mask)
                        cropped_profile = src.profile.copy()
                        cropped_profile["height"] = 128
                        cropped_profile["width"] = 128
                        cropped_profile["count"] = len(bands_to_keep)
                        cropped_profile["transform"] = rasterio.windows.transform(
                            window=rasterio.windows.Window(x, y, 128, 128),
                            transform=src.transform
                        )
                        profiles.append(cropped_profile)
    
    return processed_data, mask_data, profiles

def save_cropped_tiffs(data_list, mask_list, data_output_folder, profiles, prefix="cropped"):
    """Save each data array in the list as a GeoTIFF file using the provided profiles."""
    images_path = os.path.join(data_output_folder,'images')
    masks_path = os.path.join(data_output_folder, 'masks')
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
        
    saved_files = []
    mask_files = []
    
    for i, (data, mask, profile) in enumerate(zip(data_list, mask_list, profiles)):
        output_path = os.path.join(images_path, f"{prefix}_{i}.tif")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data.astype(np.float32))
        saved_files.append(output_path)
        
        mask_path = os.path.join(masks_path, f"{prefix}_{i}_mask.tif")
        profile["count"] = 1  # Only one band for mask
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask.astype(np.float32))
        mask_files.append(mask_path)
        
    return saved_files, mask_files

def save_filepaths_to_csv(saved_files, mask_files, csv_path):
    """Save the file paths of data and masks to a CSV."""
    with open(csv_path, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["image_filepaths", "mask_filepaths", "split"])
        
        # Determine the total number of data points
        total_data_points = len(saved_files)
        
        # Generate random assignments for each data point to one of the three splits
        split_assignments = ["train"] * int(0.8 * total_data_points) \
                          + ["val"] * int(0.1 * total_data_points) \
                          + ["test"] * int(0.1 * total_data_points)
                          
        # Shuffle the split assignments to randomize the order
        random.shuffle(split_assignments)
        
        # If the data size is not divisible by 10, append the remaining splits randomly
        while len(split_assignments) < total_data_points:
            split_assignments.append(random.choice(["train", "val", "test"]))
        
        for (data_path, mask_path, split) in zip(saved_files, mask_files, split_assignments):
            csv_writer.writerow([data_path, mask_path, split])

def main(bucket_name = 'cs325b-building-height',
    folder_path = 'data/sat_img/tx_sample_gt_2000/',
    bands_to_keep = [1, 2, 15, 16, 17],
    mask_band = [36],
    n = 2,  # Number of geotiffs to read and preprocess
    data_output_folder = "/home/spaul/group/data/full_tx2000_rgb_vv_vh"
    ):
    # Call the reading and preprocessing function
    processed_data, mask_data, profiles = read_and_preprocess_n_geotiffs_from_gcs(bucket_name, folder_path, n, bands_to_keep, mask_band)

    # Save the processed data to individual GeoTIFF files in local folders
    saved_files, mask_files = save_cropped_tiffs(processed_data, mask_data, data_output_folder, profiles)

    # Save the file paths to a CSV
    csv_file_path = os.path.join(data_output_folder, 'metadata.csv')
    save_filepaths_to_csv(saved_files, mask_files, csv_file_path)

    # print the paths of the saved files
    for file_path, mask_path in zip(saved_files, mask_files):
        print(file_path, mask_path)

if __name__ == "__main__":
    main()


