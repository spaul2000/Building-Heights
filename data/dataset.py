from typing import Dict

import numpy as np
import pandas as pd
import torch
import rasterio

class EstimationDataset(torch.utils.data.Dataset):

    def __init__(self, meta_csv: str, split: str):
        """
        Classification dataset

            Args:
                meta_csv: path to metadata csv containing paths to
                          images, masks, and associated meta data
        """
        meta_df = pd.read_csv(meta_csv)
        meta_df = meta_df[meta_df['split'] == split]
        self._image_path = list(meta_df['image_filepaths'])
        self._label = list(meta_df['mask_filepaths'])
        self._split = split


    def __len__(self) -> int:
        """Number of instances in dataset"""
        return len(self._image_path)

    def __getitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        """Retrieves and transforms tifs for image & mask of instance

            Args:
                index: specifies which image to retrieve

            Returns:
                'im': image at <index>
                'mask': ground truth labels for pixels in image

        """
        with rasterio.open(self._image_path[index]) as src:
            im_np = src.read()
            im = torch.from_numpy(im_np).float()

        with rasterio.open(self._label[index]) as src:
            label_np = src.read()
            label = torch.from_numpy(label_np).float()

        return {
            'im': im,
            'mask': label
        }
