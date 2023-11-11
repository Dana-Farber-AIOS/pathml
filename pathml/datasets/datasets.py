"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import h5py
import numpy as np
import os
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import copy
import warnings
from pathlib import Path
from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, Union
import cv2
import torchvision
from torchvision import transforms
from scipy.stats import skew
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy
from skimage.measure import regionprops
from skimage.morphology import disk
from sklearn.metrics.pairwise import euclidean_distances
from tqdm.auto import tqdm
from glob import glob

from pathml.graph.utils import HACTPairData

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
        # this part copied from h5manager.get_tile()
        tile_image = self.h5["tiles"][str(k)]["array"][:]

        # get corresponding masks if there are masks
        if "masks" in self.h5["tiles"][str(k)].keys():
            masks = {
                mask: self.h5["tiles"][str(k)]["masks"][mask][:]
                for mask in self.h5["tiles"][str(k)]["masks"]
            }
        else:
            masks = None

        labels = {
            key: val for key, val in self.h5["tiles"][str(k)]["labels"].attrs.items()
        }

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



class EntityDataset(torch.utils.data.Dataset):
    """
    Torch Geometric Dataset class for storing cell and tissue graphs. Each item returns a 
    pathml.graph.utils.HACTPairData object.

    Args:
        cell_dir (str): Path to folder containing cell graphs
        tissue_dir (str): Path to folder containing tissue graphs
        assign_dir (str): Path to folder containing assignment matrices
    """
    
    def __init__(self, cell_dir, tissue_dir, assign_dir):
        
        self.cell_graphs = glob(os.path.join(cell_dir, '*.pt') )
        self.tissue_graphs = glob(os.path.join(tissue_dir, '*.pt') )
        self.assigns = glob(os.path.join(assign_dir, '*.pt') )
        
    def __len__(self):
        return len(self.cell_graphs)
    
    def __getitem__(self, index):
        cell_graph = torch.load(self.cell_graphs[index])
        tissue_graph = torch.load(self.tissue_graphs[index])
        assignment = torch.load(self.assigns[index])
        data = HACTPairData(x_cell = cell_graph.node_features, 
                        edge_index_cell = cell_graph.edge_index, 
                        x_tissue = tissue_graph.node_features, 
                        edge_index_tissue = tissue_graph.edge_index, 
                        assignment = assignment[1,:], 
                        target = cell_graph['target'])
        return data

class InstanceMapPatchDataset(torch.utils.data.Dataset):
    """
    Create a dataset for a given image and extracted instance map with desired patches
    of (patch_size, patch_size, 3). 
    Args:
        image (np.ndarray): RGB input image.
        instance map (np.ndarray): Extracted instance map.
        entity (str): Entity to be processed. Must be one of 'cell' or 'tissue'. Defaults to 'cell'.
        patch_size (int): Desired size of patch.
        threshold (float): Threshold for processing a patch or not. 
        resize_size (int): Desired resized size to input the network. If None, no resizing is done and the
                           patches of size patch_size are provided to the network. Defaults to None.
        fill_value (Optional[int]): Value to fill outside the instance maps. Defaults to 255. 
        mean (list[float], optional): Channel-wise mean for image normalization.
        std (list[float], optional): Channel-wise std for image normalization.
        transform (Callable): Transform to apply. Defaults to None.
        with_instance_masking (bool): If pixels outside instance should be masked. Defaults to False.
    """

    def __init__(
        self,
        image,
        instance_map,
        entity = 'cell',
        patch_size = 64,
        threshold = 0.2, 
        resize_size: int = None,
        fill_value: Optional[int] = 255,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        transform: Optional[Callable] = None,
        with_instance_masking: Optional[bool] = False,
    ) -> None:

        self.image = image
        self.instance_map = instance_map
        self.entity = entity
        self.patch_size = patch_size
        self.with_instance_masking = with_instance_masking
        self.fill_value = fill_value
        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        
        self.patch_size_2 = int(self.patch_size // 2)
        
        self.image = np.pad(
            self.image,
            (
                (self.patch_size_2, self.patch_size_2),
                (self.patch_size_2, self.patch_size_2),
                (0, 0),
            ),
            mode='constant',
            constant_values = self.fill_value
        )
        self.instance_map = np.pad(
            self.instance_map,
            ((self.patch_size_2, self.patch_size_2), (self.patch_size_2, self.patch_size_2)),
            mode="constant",
            constant_values=0,
        )
        
        self.threshold = int(self.patch_size * self.patch_size * threshold)
        
        self.warning_threshold = 0.75

        basic_transforms = [transforms.ToPILImage()]
        if self.resize_size is not None:
            basic_transforms.append(transforms.Resize(self.resize_size))
        # if transform is not None:
        #     basic_transforms.append(transform)
        basic_transforms.append(transforms.ToTensor())
        if self.mean is not None and self.std is not None:
            basic_transforms.append(transforms.Normalize(self.mean, self.std))
        self.dataset_transform = transforms.Compose(basic_transforms)
        
        if self.entity == 'cell':
            self._precompute_cell()
        elif self.entity == 'tissue':
            self._precompute_tissue()
    
    def _add_patch(self, center_x: int, center_y: int, instance_index: int, region_count: int) -> None:
        """
        Extract and include patch information.
        Args:
            center_x (int): Centroid x-coordinate of the patch wrt. the instance map.
            center_y (int): Centroid y-coordinate of the patch wrt. the instance map.
            instance_index (int): Instance index to which the patch belongs.
            region_count (int): Region count indicates the location of the patch wrt. the list of patch coords.
        """
        mask = self.instance_map[
            center_y - self.patch_size_2: center_y + self.patch_size_2,
            center_x - self.patch_size_2: center_x + self.patch_size_2
        ]
        overlap = np.sum(mask == instance_index)
        if overlap > self.threshold:
            loc = [center_x - self.patch_size_2, center_y - self.patch_size_2]
            self.patch_coordinates.append(loc)
            self.patch_region_count.append(region_count)
            self.patch_instance_ids.append(instance_index)
            self.patch_overlap.append(overlap)

    def _get_patch_tissue(self, loc: list, region_id: int = None) -> np.ndarray:
        """
        Extract patch from image.
        Args:
            loc (list): Top-left (x,y) coordinate of a patch.
            region_id (int): Index of the region being processed. Defaults to None. 
        """
        min_x = loc[0]
        min_y = loc[1]
        max_x = min_x + self.patch_size
        max_y = min_y + self.patch_size

        patch = copy.deepcopy(self.image[min_y:max_y, min_x:max_x])

        if self.with_instance_masking:
            instance_mask = ~(self.instance_map[min_y:max_y, min_x:max_x] == region_id)
            patch[instance_mask, :] = self.fill_value
        return patch
    
    def _get_patch_cell(self, loc, region_id):
        min_y, min_x = loc
        patch = self.image[min_y:min_y+self.patch_size, min_x:min_x+self.patch_size,:]
        
        if self.with_instance_masking:
            instance_mask = ~(self.instance_map[min_y:min_y+self.patch_size, min_x:min_x+self.patch_size] == region_id)
            patch[instance_mask,:] = self.fill_value
        
        return patch
    
    def _precompute_cell(self):
        """Precompute instance-wise patch information for all cell instances in the input image."""
        
        self.entities = regionprops(self.instance_map)
        self.patch_coordinates = []
        self.patch_overlap = []
        self.patch_region_count = []
        self.patch_instance_ids = []

        for region_count, region in enumerate(self.entities):
            min_y, min_x, max_y, max_x = region.bbox
            coords = region.coords

            cy, cx = region.centroid
            cy, cx = int(cy), int(cx)

            coord = [cy-self.patch_size_2, cx-self.patch_size_2]

            instance_mask = self.instance_map[coord[0]:coord[0]+self.patch_size, coord[1]:coord[1]+self.patch_size]
            overlap = np.sum(instance_mask == region.label)
            if overlap >= self.threshold:
                self.patch_coordinates.append(coord)
                self.patch_region_count.append(region_count)
                self.patch_instance_ids.append(region.label)
                self.patch_overlap.append(overlap)

    def _precompute_tissue(self):
        """Precompute instance-wise patch information for all tissue instances in the input image."""
        
        self.patch_coordinates = []
        self.patch_region_count = []
        self.patch_instance_ids = []
        self.patch_overlap = []
        
        self.entities = regionprops(self.instance_map)
        self.stride = self.patch_size
        
        for region_count, region in enumerate(self.entities):

            # Extract centroid
            center_y, center_x = region.centroid
            center_x = int(round(center_x))
            center_y = int(round(center_y))

            # Extract bounding box
            min_y, min_x, max_y, max_x = region.bbox

            # Extract patch information around the centroid patch 
            # quadrant 1 (includes centroid patch)
            y_ = copy.deepcopy(center_y)
            while y_ >= min_y:
                x_ = copy.deepcopy(center_x)
                while x_ >= min_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ -= self.stride
                y_ -= self.stride

            # quadrant 4
            y_ = copy.deepcopy(center_y)
            while y_ >= min_y:
                x_ = copy.deepcopy(center_x) + self.stride
                while x_ <= max_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ += self.stride
                y_ -= self.stride

            # quadrant 2
            y_ = copy.deepcopy(center_y) + self.stride
            while y_ <= max_y:
                x_ = copy.deepcopy(center_x)
                while x_ >= min_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ -= self.stride
                y_ += self.stride

            # quadrant 3
            y_ = copy.deepcopy(center_y) + self.stride
            while y_ <= max_y:
                x_ = copy.deepcopy(center_x) + self.stride
                while x_ <= max_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ += self.stride
                y_ += self.stride

    def _warning(self):
        """Check patch coverage statistics to identify if provided patch size includes too much background."""
        self.patch_overlap = np.array(self.patch_overlap) / (
            self.patch_size * self.patch_size
        )
        if np.mean(self.patch_overlap) < self.warning_threshold:
            warnings.warn("Provided patch size is large")
            warnings.warn(
                "Suggestion: Reduce patch size to include relevant context.")
    
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Loads an image for a given patch index.
        Args:
            index (int): Patch index.
        Returns:
            Tuple[torch.Tensor, int]: image as tensor, instance_index.
        """
        
        if self.entity == 'cell':
            patch = self._get_patch_cell(self.patch_coordinates[index],
                                           self.patch_instance_ids[index])
        elif self.entity == 'tissue':
            patch = self._get_patch_tissue(self.patch_coordinates[index],
                                           self.patch_instance_ids[index])
        
        patch = self.dataset_transform(patch)
        return patch, self.patch_region_count[index]
        

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        Returns:
            int: Length of the dataset
        """
        return len(self.patch_coordinates)
