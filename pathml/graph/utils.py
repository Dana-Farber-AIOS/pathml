import importlib
import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from skimage.measure import label, regionprops
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_edge_index, to_torch_csr_tensor

MIN_NR_PIXELS = 50000
MAX_NR_PIXELS = 50000000


class Graph(Data):
    """Constructs pytorch-geometric data object for saving and loading"""

    def __init__(self, node_centroids, node_features, edge_index, node_labels, target):
        super().__init__()
        self.node_centroids = node_centroids
        self.node_features = node_features
        self.edge_index = edge_index
        self.node_labels = node_labels
        self.target = target

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return self.node_features.size(0)
        elif key == "target":
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)


class HACTPairData(Data):
    """Constructs pytorch-geometric data object for handling both cell and tissue data"""

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


def dynamic_import_from(source_file: str, class_name: str):
    """Do a from source_file import class_name dynamically

    Args:
        source_file (str): Where to import from
        class_name (str): What to import
    Returns:
        Any: The class to be imported
    """
    module = importlib.import_module(source_file)
    return getattr(module, class_name)


def _valid_image(nr_pixels):
    """
    Checks if image does not exceed maximum number of pixels or exceeds minimum number of pixels.

    Args:
        nr_pixels (int): Number of pixels in given image
    """

    if nr_pixels > MIN_NR_PIXELS and nr_pixels < MAX_NR_PIXELS:
        return True
    return False


def plot_graph_on_image(graph, image):
    """
    Plots a given graph on the original WSI image

    Args:
        graph (torch.tensor): Graph as an sparse edge index
        image (numpy.array): Input image
    """

    from torch_geometric.utils.convert import to_networkx

    pos = graph.node_centroids.numpy()
    G = to_networkx(graph, to_undirected=True)
    plt.imshow(image)
    nx.draw(G, pos, node_size=25)
    plt.show()


def _exists(cg_out, tg_out, assign_out, overwrite):
    """
    Checks if given input files exist or not

    Args:
        cg_out (str): Cell graph file
        tg_out (str): Tissue graph file
        assign_out (str): Assignment matrix file
        overwrite (bool): Whether to overwrite files or not. If true, this function return false and files are
                          overwritten.
    """

    if overwrite:
        return False
    else:
        if (
            os.path.isfile(cg_out)
            and os.path.isfile(tg_out)
            and os.path.isfile(assign_out)
        ):
            return True
        return False


def get_full_instance_map(wsi, patch_size, mask_name="cell"):
    """
    Generates and returns the normalized image, cell instance map and cell centroids from pathml SlideData object

    Args:
        wsi (pathml.core.SlideData): Normalized WSI object with detected cells in the 'masks' slot
        patch_size (int): Patch size used for cell detection
        mask_name (str): Name of the mask slot storing the detected cells. Defaults to 'cell'.
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
        matrix (bool): Whether to return in a matrix format. If True, returns a N*L matrix where N is the number of low-level instances and L is the number of high-level instances. If False, returns this mapping in sparse format. Defaults to False.
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


def compute_histogram(input_array: np.ndarray, nr_values: int) -> np.ndarray:
    """Calculates a histogram of a matrix of the values from 0 up to (excluding) nr_values
    Args:
        x (np.array): Input tensor
        nr_values (int): Possible values. From 0 up to (exclusing) nr_values.
    Returns:
        np.array: Output tensor
    """
    output_array = np.empty(nr_values, dtype=int)
    for i in range(nr_values):
        output_array[i] = (input_array == i).sum()
    return output_array


def two_hop(edge_index, num_nodes):
    """Calculates the two-hop graph
    Args:
        edge_index (torch.tensor): The edge index in sparse form of the graph
        num_nodes (int): maximum number of nodes
    Returns:
        torch.tensor: Output edge index tensor
    """
    adj = to_torch_csr_tensor(edge_index, size=(num_nodes, num_nodes))
    edge_index2, _ = to_edge_index(adj @ adj)
    edge_index2, _ = remove_self_loops(edge_index2)
    edge_index = torch.cat([edge_index, edge_index2], dim=1)
    return edge_index
