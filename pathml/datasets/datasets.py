"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import copy
import os
import warnings
from glob import glob

import h5py
import numpy as np
import torch
from skimage.measure import regionprops
from skimage.transform import resize

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
    Torch Geometric Dataset class for storing cell or tissue graphs. Each item returns a
    pathml.graph.utils.HACTPairData object.

    Args:
        cell_dir (str): Path to folder containing cell graphs
        tissue_dir (str): Path to folder containing tissue graphs
        assign_dir (str): Path to folder containing assignment matrices
    """

    def __init__(self, cell_dir=None, tissue_dir=None, assign_dir=None):
        self.cell_dir = cell_dir
        self.tissue_dir = tissue_dir
        self.assign_dir = assign_dir

        if self.cell_dir is not None:
            if not os.path.exists(cell_dir):
                raise FileNotFoundError(f"Directory not found: {self.cell_dir}")
            self.cell_graphs = glob(os.path.join(cell_dir, "*.pt"))

        if self.tissue_dir is not None:
            if not os.path.exists(tissue_dir):
                raise FileNotFoundError(f"Directory not found: {self.tissue_dir}")
            self.tissue_graphs = glob(os.path.join(tissue_dir, "*.pt"))

        if self.assign_dir is not None:
            if not os.path.exists(assign_dir):
                raise FileNotFoundError(f"Directory not found: {self.assign_dir}")
            self.assigns = glob(os.path.join(assign_dir, "*.pt"))

    def __len__(self):
        return len(self.cell_graphs)

    def __getitem__(self, index):

        target = None

        # Load cell graphs, tissue graphs and assignments if they are provided
        if self.cell_dir is not None:
            cell_graph = torch.load(self.cell_graphs[index])
            if hasattr(cell_graph, "target"):
                target = cell_graph["target"]
            else:
                target = None

        if self.tissue_dir is not None:
            tissue_graph = torch.load(self.tissue_graphs[index])
            if hasattr(tissue_graph, "target"):
                target = tissue_graph["target"]
            else:
                target = None

        if self.assign_dir is not None:
            assignment = torch.load(self.assigns[index])

        # Create pathml.graph.utils.HACTPairData object with prvided objects
        data = HACTPairData(
            x_cell=cell_graph.node_features if self.cell_dir is not None else None,
            edge_index_cell=cell_graph.edge_index
            if self.cell_dir is not None
            else None,
            x_tissue=tissue_graph.node_features
            if self.tissue_dir is not None
            else None,
            edge_index_tissue=tissue_graph.edge_index
            if self.tissue_dir is not None
            else None,
            assignment=assignment[1, :] if self.assign_dir is not None else None,
            target=target,
        )
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
        threshold (float): Minimum threshold for processing a patch or not.
        resize_size (int): Desired resized size to input the network. If None, no resizing is done and the
                           patches of size patch_size are provided to the network. Defaults to None.
        fill_value (Optional[int]): Value to fill outside the instance maps. Defaults to 255.
        mean (list[float], optional): Channel-wise mean for image normalization.
        std (list[float], optional): Channel-wise std for image normalization.
        with_instance_masking (bool): If pixels outside instance should be masked. Defaults to False.
    """

    def __init__(
        self,
        image,
        instance_map,
        entity="cell",
        patch_size=64,
        threshold=0.2,
        resize_size=None,
        fill_value=255,
        mean=None,
        std=None,
        with_instance_masking=False,
    ):

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
            mode="constant",
            constant_values=self.fill_value,
        )
        self.instance_map = np.pad(
            self.instance_map,
            (
                (self.patch_size_2, self.patch_size_2),
                (self.patch_size_2, self.patch_size_2),
            ),
            mode="constant",
            constant_values=0,
        )

        self.threshold = int(self.patch_size * self.patch_size * threshold)
        self.warning_threshold = 0.50

        try:
            from torchvision import transforms

            self.use_torchvision = True
        except ImportError:
            print(
                "Torchvision is not installed, using base modules for resizing patches and skipping normalization"
            )
            self.use_torchvision = False

        if self.use_torchvision:
            basic_transforms = [transforms.ToPILImage()]
            if self.resize_size is not None:
                basic_transforms.append(transforms.Resize(self.resize_size))
            basic_transforms.append(transforms.ToTensor())
            if self.mean is not None and self.std is not None:
                basic_transforms.append(transforms.Normalize(self.mean, self.std))
            self.dataset_transform = transforms.Compose(basic_transforms)

        if self.entity not in ["cell", "tissue"]:
            raise ValueError(
                "Invalid value for entity. Expected 'cell' or 'tissue', got '{}'.".format(
                    self.entity
                )
            )

        if self.entity == "cell":
            self._precompute_cell()
        elif self.entity == "tissue":
            self._precompute_tissue()

        self._warning()

    def _add_patch(self, center_x, center_y, instance_index, region_count):
        """Extract and include patch information."""

        # Get a patch for each entity in the instance map
        mask = self.instance_map[
            center_y - self.patch_size_2 : center_y + self.patch_size_2,
            center_x - self.patch_size_2 : center_x + self.patch_size_2,
        ]

        # Check the overlap between the extracted patch and the entity
        overlap = np.sum(mask == instance_index)

        # Add patch coordinates if overlap is greated than threshold
        if overlap > self.threshold:
            loc = [center_x - self.patch_size_2, center_y - self.patch_size_2]
            self.patch_coordinates.append(loc)
            self.patch_region_count.append(region_count)
            self.patch_instance_ids.append(instance_index)
            self.patch_overlap.append(overlap)

    def _get_patch_tissue(self, loc, region_id=None):
        """Extract tissue patches from image."""

        # Get bounding box of given location
        min_x = loc[0]
        min_y = loc[1]
        max_x = min_x + self.patch_size
        max_y = min_y + self.patch_size

        patch = copy.deepcopy(self.image[min_y:max_y, min_x:max_x])

        # Fill background pixels with instance masking value
        if self.with_instance_masking:
            instance_mask = ~(self.instance_map[min_y:max_y, min_x:max_x] == region_id)
            patch[instance_mask, :] = self.fill_value
        return patch

    def _get_patch_cell(self, loc, region_id):
        """Extract cell patches from image."""

        # Get bounding box of given location
        min_y, min_x = loc
        patch = self.image[
            min_y : min_y + self.patch_size, min_x : min_x + self.patch_size, :
        ]

        # Fill background pixels with instance masking value
        if self.with_instance_masking:
            instance_mask = ~(
                self.instance_map[
                    min_y : min_y + self.patch_size, min_x : min_x + self.patch_size
                ]
                == region_id
            )
            patch[instance_mask, :] = self.fill_value

        return patch

    def _precompute_cell(self):
        """Precompute instance-wise patch information for all cell instances in the input image."""

        # Get location of all entities from the instance map
        self.entities = regionprops(self.instance_map)
        self.patch_coordinates = []
        self.patch_overlap = []
        self.patch_region_count = []
        self.patch_instance_ids = []

        # Get coordinates for all entities and add them to the pile
        for region_count, region in enumerate(self.entities):
            min_y, min_x, max_y, max_x = region.bbox

            cy, cx = region.centroid
            cy, cx = int(cy), int(cx)

            coord = [cy - self.patch_size_2, cx - self.patch_size_2]

            instance_mask = self.instance_map[
                coord[0] : coord[0] + self.patch_size,
                coord[1] : coord[1] + self.patch_size,
            ]
            overlap = np.sum(instance_mask == region.label)
            if overlap >= self.threshold:
                self.patch_coordinates.append(coord)
                self.patch_region_count.append(region_count)
                self.patch_instance_ids.append(region.label)
                self.patch_overlap.append(overlap)

    def _precompute_tissue(self):
        """Precompute instance-wise patch information for all tissue instances in the input image."""

        # Get location of all entities from the instance map
        self.patch_coordinates = []
        self.patch_region_count = []
        self.patch_instance_ids = []
        self.patch_overlap = []

        self.entities = regionprops(self.instance_map)
        self.stride = self.patch_size

        # Get coordinates for all entities and add them to the pile
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
            warnings.warn("Suggestion: Reduce patch size to include relevant context.")

    def __getitem__(self, index):
        """Loads an image for a given patch index."""

        if self.entity == "cell":
            patch = self._get_patch_cell(
                self.patch_coordinates[index], self.patch_instance_ids[index]
            )
        elif self.entity == "tissue":
            patch = self._get_patch_tissue(
                self.patch_coordinates[index], self.patch_instance_ids[index]
            )
        else:
            raise ValueError(
                "Invalid value for entity. Expected 'cell' or 'tissue', got '{}'.".format(
                    self.entity
                )
            )

        if self.use_torchvision:
            patch = self.dataset_transform(patch)
        else:
            patch = patch / 255.0 if patch.max() > 1 else patch
            patch = resize(patch, (self.resize_size, self.resize_size))
            patch = torch.from_numpy(patch).permute(2, 0, 1).float()

        return patch, self.patch_region_count[index]

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.patch_coordinates)
