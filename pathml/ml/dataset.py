"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import torch
import h5py


class TileDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for h5path files

    Each item is a tuple of (``tile_image``, ``tile_masks``, ``tile_labels``, ``slide_labels``) where:

        - ``tile_image`` is a torch.Tensor of shape (C, H, W) or (T, Z, C, H, W)
        - ``tile_masks`` is a torch.Tensor of shape (n_masks, tile_height, tile_width)
        - ``tile_labels`` is a dict
        - ``slide_labels`` is a dict

    This is designed to be wrapped in a PyTorch DataLoader for feeding tiles into ML models.

    Note that label dictionaries are not standardized, as users are free to store whatever labels they want.
    For that reason, PyTorch cannot automatically stack labels into batches.
    When creating a DataLoader from a TileDataset, it may therefore be necessary to create a custom ``collate_fn`` to
    specify how to create batches of labels. See: https://discuss.pytorch.org/t/how-to-use-collate-fn/27181

    Args:
        file_path (str): Path to .h5path file on disk
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.h5 = None
        with h5py.File(self.file_path, "r") as file:
            self.tile_shape = eval(file["tiles"].attrs["tile_shape"])
            self.tile_keys = list(file["tiles"].keys())
            self.dataset_len = len(self.tile_keys)
            self.slide_level_labels = {
                key: val
                for key, val in file["fields"]["labels"].attrs.items()
                if val is not None
            }

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, ix):
        if self.h5 is None:
            self.h5 = h5py.File(self.file_path, "r")

        k = self.tile_keys[ix]
        coords = list(eval(k))
        ### this part copied from h5manager.get_tile()

        # impute missing dimensions from self.h5["tiles"].attrs["tile_shape"]
        if len(self.tile_shape) > len(coords):
            shape = list(self.tile_shape)
            coords = coords + [0] * len(shape[len(coords) - 1 :])
        tiler = [
            slice(coords[i], coords[i] + self.tile_shape[i])
            for i in range(len(self.tile_shape))
        ]
        tile_image = self.h5["array"][tuple(tiler)][:]

        # get corresponding masks if there are masks
        if "masks" in self.h5.keys():
            try:
                masks = {
                    mask: self.h5["masks"][mask][tuple(tiler)][:]
                    for mask in self.h5["masks"]
                }
            except (ValueError, TypeError):
                # mask may have fewer dimensions, e.g. a 2-d mask for a 3-d image.
                masks = {}
                for mask in self.h5["masks"]:
                    n_dim_mask = self.h5["masks"][mask].ndim
                    mask_tiler = tuple(tiler)[0:n_dim_mask]
                    masks[mask] = self.h5["masks"][mask][mask_tiler][:]
        else:
            masks = None

        labels = {key: val for key, val in self.h5["tiles"][k]["labels"].attrs.items()}

        if tile_image.ndim == 3:
            # swap axes from HWC to CHW for pytorch
            im = tile_image.transpose(2, 0, 1)
        elif tile_image.ndim == 5:
            # in this case, we assume that we have XYZCT channel order (OME-TIFF)
            # so we swap axes to TCZYX for batching
            im = tile_image.transpose(4, 3, 2, 1, 0)
        else:
            raise NotImplementedError(
                f"tile image has shape {tile_image.shape}. Expecting an image with 3 dims (HWC) or 5 dims (XYZCT)"
            )

        masks = np.stack(list(masks.values()), axis=0) if masks else None

        return im, masks, labels, self.slide_level_labels
