"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import math

import numpy as np
import torch
from skimage.measure import label, regionprops
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_edge_index, to_torch_csr_tensor

MIN_NR_PIXELS = 50000
MAX_NR_PIXELS = 50000000


class Graph(Data):
    """Constructs pytorch-geometric data object for saving and loading

    Args:
        node_centroids (torch.tensor): Coordinates of the centers of each entity (cell or tissue) in the graph
        node_features (torch.tensor): Computed features of each entity (cell or tissue) in the graph
        edge_index (torch.tensor): Edge index in sparse format between nodes in the graph
        node_labels  (torch.tensor): Node labels of each entity (cell or tissue) in the graph. Defaults to None.
        target (torch.tensor): Target label if used in a supervised setting. Defaults to None.
    """

    def __init__(
        self,
        node_centroids,
        edge_index,
        node_features=None,
        node_labels=None,
        edge_features=None,
        target=None,
    ):
        super().__init__()
        self.node_centroids = node_centroids
        self.node_features = node_features
        self.edge_index = edge_index
        self.node_labels = node_labels
        self.target = target
        self.edge_features = edge_features

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return self.node_centroids.size(0)
        elif key == "target":
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)


class HACTPairData(Data):
    """Constructs pytorch-geometric data object for handling both cell and tissue data

    Args:
        x_cell (torch.tensor): Computed features of each cell in the graph
        edge_index_cell (torch.tensor): Edge index in sparse format between nodes in the cell graph
        x_tissue (torch.tensor): Computed features of each tissue in the graph
        edge_index_tissue (torch.tensor): Edge index in sparse format between nodes in the tissue graph
        assignment (torch.tensor): Assigment matrix that contains mapping between cells and tissues.
        target (torch.tensor): Target label if used in a supervised setting.
    """

    def __init__(
        self, x_cell, edge_index_cell, x_tissue, edge_index_tissue, assignment, target
    ):
        super().__init__()
        self.x_cell = x_cell
        self.edge_index_cell = edge_index_cell

        self.x_tissue = x_tissue
        self.edge_index_tissue = edge_index_tissue

        self.assignment = assignment
        self.target = target

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_cell":
            return self.x_cell.size(0)
        if key == "edge_index_tissue":
            return self.x_tissue.size(0)
        elif key == "assignment":
            return self.x_tissue.size(0)
        elif key == "target":
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)


def get_full_instance_map(wsi, patch_size, mask_name="cell"):
    """
    Generates and returns the normalized image, cell instance map and cell centroids from pathml SlideData object

    Args:
        wsi (pathml.core.SlideData): Normalized WSI object with detected cells in the 'masks' slot
        patch_size (int): Patch size used for cell detection
        mask_name (str): Name of the mask slot storing the detected cells. Defaults to 'cell'.

    Returns:
        The image in np.unint8 format, the instance map for the entity and the instance centroids for each entity in
        the instance map as numpy arrays.
    """

    x = math.ceil(wsi.shape[0] / patch_size) * patch_size
    y = math.ceil(wsi.shape[1] / patch_size) * patch_size
    image_norm = np.zeros((x, y, 3))
    instance_map = np.zeros((x, y))
    for tile in wsi.tiles:
        tx, ty = tile.coords
        image_norm[tx : tx + patch_size, ty : ty + patch_size] = tile.image
        instance_map[tx : tx + patch_size, ty : ty + patch_size] = tile.masks[
            mask_name
        ][:, :, 0]
    image_norm = image_norm[: wsi.shape[0], : wsi.shape[1], :]
    instance_map = instance_map[: wsi.shape[0], : wsi.shape[1]]
    label_instance_map = label(instance_map)
    regions = regionprops(label_instance_map)
    instance_centroids = np.empty((len(regions), 2))
    for i, region in enumerate(regions):
        center_y, center_x = region.centroid  # row, col
        center_x = int(round(center_x))
        center_y = int(round(center_y))
        instance_centroids[i, 0] = center_x
        instance_centroids[i, 1] = center_y
    return image_norm.astype("uint8"), label_instance_map, instance_centroids


def build_assignment_matrix(low_level_centroids, high_level_map, matrix=False):
    """
    Builds an assignment matrix/mapping between low-level centroid locations and a high-level segmentation map

    Args:
        low_level_centroids (numpy.array): The low-level centroid coordinates in x-y plane
        high-level map (numpy.array): The high-level map returned from regionprops
        matrix (bool): Whether to return in a matrix format. If True, returns a N*L matrix where N is the number of low-level
            instances and L is the number of high-level instances. If False, returns this mapping in sparse format.
            Defaults to False.

    Returns:
        The assignment matrix as a numpy array.
    """

    low_level_centroids = low_level_centroids.astype(int)
    low_to_high = high_level_map[
        low_level_centroids[:, 1], low_level_centroids[:, 0]
    ].astype(int)
    high_instance_ids = np.sort(np.unique(np.ravel(high_level_map))).astype(int)
    if 0 in high_instance_ids:
        high_instance_ids = np.delete(high_instance_ids, 0)
    assignment_matrix = np.zeros((low_level_centroids.shape[0], len(high_instance_ids)))
    assignment_matrix[np.arange(low_to_high.size), low_to_high - 1] = 1
    if not matrix:
        sparse_matrix = np.nonzero(assignment_matrix)
        return np.array(sparse_matrix)
    return assignment_matrix


def two_hop(edge_index, num_nodes):
    """Calculates the two-hop graph.
    Args:
        edge_index (torch.tensor): The edge index in sparse form of the graph.
        num_nodes (int): maximum number of nodes.
    Returns:
        torch.tensor: Output edge index tensor.
    """
    adj = to_torch_csr_tensor(edge_index, size=(num_nodes, num_nodes))
    edge_index2, _ = to_edge_index(adj @ adj)
    edge_index2, _ = remove_self_loops(edge_index2)
    edge_index = torch.cat([edge_index, edge_index2], dim=1)
    return edge_index
