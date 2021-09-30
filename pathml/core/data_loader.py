"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import torch
import h5py

from pathml.core.h5managers import h5pathManager


class TileDataset(torch.utils.data.Dataset, h5pathManager):
    """
    Dataset for h5path files

    Each item is a tuple of (tile_image, tile_masks, tile_labels, slide_labels)
    Where:
        tile_image is a torch.Tensor of shape (n_channels, tile_height, tile_width)
        tile_masks is a torch.Tensor of shape (n_masks, tile_height, tile_width)
        tile_labels is a dict
        slide_labels is a dict
    """

    # inherits from h5pathManager class so that we can use `get_tile()` method instead of reimplementing here
    def __init__(self, path):
        self.file_path = path
        self.h5 = None
        with h5py.File(self.file_path, "r") as file:
            self.tile_shape = eval(file["tiles"].attrs["tile_shape"])
            self.tile_keys = list(file["tiles"].keys())
            self.dataset_len = len(self.tile_keys)
            self.slide_level_labels = {
                key: val
                for key, val in self.h5["fields"]["labels"].attrs.items()
                if val is not None
            }

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, ix):
        if self.h5 is None:
            self.h5 = h5py.File(self.file_path, "r")
        tile = self.get_tile(item=ix)

        im = torch.from_numpy(tile.image)
        mask = torch.from_numpy(tile.masks)
        # swap axes from HWC to CHW for pytorch
        im = im.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)

        return im, mask, tile.labels, self.slide_level_labels
