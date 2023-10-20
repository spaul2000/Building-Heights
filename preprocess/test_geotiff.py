import rasterio
import numpy as np
import matplotlib.pyplot as plt


def stitch_crops_together(crop_files):
    """Stitch together cropped GeoTIFFs into a single image array."""
    
    # Read the first crop to get metadata info
    with rasterio.open(crop_files[0]) as src:
        meta = src.meta.copy()
        full_width = meta['width'] * int(np.sqrt(len(crop_files)))
        full_height = meta['height'] * int(np.sqrt(len(crop_files)))
        full_image = np.zeros((meta['count'], full_height, full_width))
    
    # Populate the full_image array with data from the cropped images
    for crop_file in crop_files:
        with rasterio.open(crop_file) as src:
            x, y = src.xy(0, 0, offset='ul')  # get the upper-left x,y values
            col_off, row_off = ~meta['transform'] * (x, y)
            col_off, row_off = int(round(col_off)), int(round(row_off))

            print(f"Placing crop at column offset: {col_off}, row offset: {row_off}")
            
            full_image[:, row_off:row_off+src.height, col_off:col_off+src.width] = src.read()
    
    return full_image

# Example usage:
crop_files = [f"./data/mean_crops/cropped_{i}.tif" for i in range(9)]  # Adjust based on your files
stitched_image = stitch_crops_together(crop_files)

plt.imshow(stitched_image[0])  # Display the first band of the stitched image
plt.colorbar()
plt.title("Stitched Image")
plt.show()
plt.savefig('preprocess/test.png')

