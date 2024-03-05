"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import math
from abc import abstractmethod

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import skimage
import torch

if skimage.__version__ < "0.20.0":  # pragma: no cover
    from skimage.future import graph
else:
    from skimage import graph

from skimage.color.colorconv import rgb2hed
from skimage.measure import regionprops
from skimage.segmentation import slic
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils.convert import from_networkx, to_networkx

from pathml.graph.utils import Graph, two_hop


class GraphFeatureExtractor:
    """
    Extracts features from a networkx graph object.

    Args:
        use_weight (bool, optional): Whether to use edge weights for feature computation. Defaults to False.
        alpha (float, optional): Alpha value for personalized page-rank. Defaults to 0.85.

    Returns:
        Dictionary of keys as feature type and values as features
    """

    def __init__(self, use_weight=False, alpha=0.85):
        self.use_weight = use_weight
        self.feature_dict = {}
        self.alpha = alpha

    def get_stats(self, dct, prefix="add_pre"):
        local_dict = {}
        lst = list(dct.values())
        local_dict[f"{prefix}_mean"] = np.mean(lst)
        local_dict[f"{prefix}_median"] = np.median(lst)
        local_dict[f"{prefix}_max"] = np.max(lst)
        local_dict[f"{prefix}_min"] = np.min(lst)
        local_dict[f"{prefix}_sum"] = np.sum(lst)
        local_dict[f"{prefix}_std"] = np.std(lst)
        return local_dict

    def process(self, G):
        if self.use_weight:
            if "weight" in list(list(G.edges(data=True))[0][-1].keys()):
                weight = "weight"
            else:  # pragma: no cover
                raise ValueError(
                    "No edge attribute called 'weight' when use_weight is True"
                )
        else:
            weight = None

        self.feature_dict["diameter"] = nx.diameter(G)
        self.feature_dict["radius"] = nx.radius(G)
        self.feature_dict["assortativity_degree"] = nx.degree_assortativity_coefficient(
            G
        )
        self.feature_dict["density"] = nx.density(G)
        self.feature_dict["transitivity_undir"] = nx.transitivity(G)

        self.feature_dict.update(self.get_stats(nx.hits(G)[0], prefix="hubs"))
        self.feature_dict.update(self.get_stats(nx.hits(G)[1], prefix="authorities"))
        self.feature_dict.update(
            self.get_stats(nx.constraint(G, weight=weight), prefix="constraint")
        )
        self.feature_dict.update(self.get_stats(nx.core_number(G), prefix="coreness"))
        self.feature_dict.update(
            self.get_stats(
                nx.eigenvector_centrality(G, weight=weight, max_iter=500),
                prefix="egvec_centr",
            )
        )
        self.feature_dict.update(
            self.get_stats(
                {node: val for (node, val) in G.degree(weight=weight)}, prefix="degree"
            )
        )
        self.feature_dict.update(
            self.get_stats(
                nx.pagerank(G, alpha=self.alpha), prefix="personalized_pgrank"
            )
        )

        return self.feature_dict


class BaseGraphBuilder:
    """Base interface class for graph building.

    Args:
        nr_annotation_classes (int): Number of classes in annotation. Used only if setting node labels.
        annotation_background_class (int): Background class label in annotation. Used only if setting node labels.
        add_loc_feats (bool): Flag to include location-based features (ie normalized centroids) in node feature
                              representation. Defaults to False.
        return_networkx (bool): Whether to return as a networkx graph object. Deafults to returning a Pytorvh Geometric
                                Data object.

    References:
        [1] https://github.com/BiomedSciAI/histocartography/tree/main
        [2] Jaume, G., Pati, P., Anklin, V., Foncubierta, A. and Gabrani, M., 2021, September.
        Histocartography: A toolkit for graph analytics in digital pathology.
        In MICCAI Workshop on Computational Pathology (pp. 117-128). PMLR.
    """

    def __init__(
        self,
        nr_annotation_classes: int = 5,
        annotation_background_class=None,
        add_loc_feats=False,
        return_networkx=False,
        **kwargs,
    ):
        """Base Graph Builder constructor."""
        self.nr_annotation_classes = nr_annotation_classes
        self.annotation_background_class = annotation_background_class
        self.add_loc_feats = add_loc_feats
        self.return_networkx = return_networkx
        super().__init__(**kwargs)

    def process(  # type: ignore[override]
        self, instance_map, features=None, annotation=None, target=None
    ):
        """Generates a graph from a given instance_map and features"""
        # add nodes
        self.num_nodes = features.shape[0]

        # add image size as graph data
        image_size = (instance_map.shape[1], instance_map.shape[0])  # (x, y)

        # get instance centroids
        self.centroids = self._get_node_centroids(instance_map)

        # add node content
        if features is not None:
            node_features = self._compute_node_features(features, image_size)
        else:
            node_features = None

        if annotation is not None:
            node_labels = self._set_node_labels(annotation, self.centroids.shape[0])
        else:
            node_labels = None

        # build edges
        edges = self._build_topology(instance_map)

        # compute edge features
        edge_features = self._compute_edge_features(edges)

        # make torch geometric data object
        graph = Graph(
            node_centroids=self.centroids,
            node_features=node_features,
            edge_index=edges,
            edge_features=edge_features,
            node_labels=node_labels,
            target=torch.tensor(target) if target is not None else None,
        )

        if self.return_networkx:

            node_attrs = [
                "node_centroids",
                "node_features" if node_features is not None else None,
                "node_labels" if node_labels is not None else None,
            ]
            node_attrs = list(filter(lambda item: item is not None, node_attrs))

            edge_attrs = ["edge_features" if edge_features is not None else None]
            edge_attrs = list(filter(lambda item: item is not None, edge_attrs))

            graph_attrs = ["target" if target is not None else None]
            graph_attrs = list(filter(lambda item: item is not None, graph_attrs))

            return to_networkx(
                graph,
                node_attrs=node_attrs,
                edge_attrs=edge_attrs,
                graph_attrs=graph_attrs,
            )
        else:
            return graph

    def process_with_centroids(
        self, centroids, features=None, image_size=None, annotation=None, target=None
    ):
        """Generates a graph from a given node centroids and features"""

        self.centroids = centroids

        # add nodes
        self.num_nodes = self.centroids.shape[0]

        # add node content
        if features is not None:
            node_features = self._compute_node_features(features, image_size)
        else:
            node_features = None

        if annotation is not None:
            node_labels = self._set_node_labels(annotation, self.num_nodes)
        else:
            node_labels = None

        # build edges
        edges = self._build_topology(None)

        # compute edge features
        edge_features = self._compute_edge_features(edges)

        # make torch geometric data object
        graph = Graph(
            node_centroids=self.centroids,
            node_features=node_features,
            edge_index=edges,
            node_labels=node_labels,
            target=torch.tensor(target) if target is not None else None,
        )

        if self.return_networkx:
            node_attrs = [
                "node_centroids",
                "node_features" if node_features is not None else None,
                "node_labels" if node_labels is not None else None,
            ]
            node_attrs = list(filter(lambda item: item is not None, node_attrs))

            edge_attrs = ["edge_features" if edge_features is not None else None]
            edge_attrs = list(filter(lambda item: item is not None, edge_attrs))

            graph_attrs = ["target" if target is not None else None]
            graph_attrs = list(filter(lambda item: item is not None, graph_attrs))

            return to_networkx(
                graph,
                node_attrs=node_attrs,
                edge_attrs=edge_attrs,
                graph_attrs=graph_attrs,
            )
        else:
            return graph

    def _get_node_centroids(self, instance_map):
        """Get the centroids of the graphs"""
        regions = regionprops(instance_map)
        centroids = np.empty((len(regions), 2))
        for i, region in enumerate(regions):
            center_y, center_x = region.centroid  # (y, x)
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            centroids[i, 0] = center_x
            centroids[i, 1] = center_y
        return torch.tensor(centroids)

    def _compute_node_features(self, features, image_size):
        """Set the provided node features"""
        if not torch.is_tensor(features):
            features = torch.FloatTensor(features)
        if not self.add_loc_feats:
            return features
        elif self.add_loc_feats and image_size is not None:
            # compute normalized centroid features

            normalized_centroids = torch.empty_like(self.centroids)  # (x, y)
            normalized_centroids[:, 0] = self.centroids[:, 0] / image_size[0]
            normalized_centroids[:, 1] = self.centroids[:, 1] / image_size[1]

            if features.ndim == 3:
                normalized_centroids = normalized_centroids.unsqueeze(dim=1).repeat(
                    1, features.shape[1], 1
                )
                concat_dim = 2
            elif features.ndim == 2:
                concat_dim = 1

            concat_features = torch.cat(
                (features, normalized_centroids),
                dim=concat_dim,
            )
            return concat_features
        else:  # pragma: no cover
            raise ValueError(
                "Please provide image size to add the normalized centroid to the node features."
            )

    @abstractmethod
    def _set_node_labels(self, instance_map, annotation):
        """Set the node labels of the graphs"""

    @abstractmethod
    def _compute_edge_features(self, edges):
        """Set the provided edge features"""

    @abstractmethod
    def _build_topology(self, instance_map):
        """Generate the graph topology from the provided instance_map"""


class KNNGraphBuilder(BaseGraphBuilder):
    """
    k-Nearest Neighbors Graph class for graph building.

    Args:
        k (int, optional): Number of neighbors. Defaults to 5.
        thresh (int, optional): Maximum allowed distance between 2 nodes. Defaults to None (no thresholding).

    Returns:
        A pathml.graph.utils.Graph object containing node and edge information.

    References:
        [1] https://github.com/BiomedSciAI/histocartography/tree/main
        [2] Jaume, G., Pati, P., Anklin, V., Foncubierta, A. and Gabrani, M., 2021, September.
        Histocartography: A toolkit for graph analytics in digital pathology.
        In MICCAI Workshop on Computational Pathology (pp. 117-128). PMLR.
    """

    def __init__(self, k=5, thresh=None, **kwargs):
        """Create a graph builder that uses the (thresholded) kNN algorithm to define the graph topology."""

        self.k = k
        self.thresh = thresh
        super().__init__(**kwargs)

    def _set_node_labels(self, annotation, num_nodes):
        """Set the node labels of the graphs using annotation"""
        assert (
            annotation.shape[0] == num_nodes
        ), "Number of annotations do not match number of nodes"
        return annotation

    def _build_topology(self, instance_map):
        """Build topology using (thresholded) kNN"""

        # build kNN adjacency
        adjacency = kneighbors_graph(
            self.centroids,
            self.k,
            mode="distance",
            include_self=False,
            metric="euclidean",
        ).toarray()

        # filter edges that are too far (ie larger than thresh)
        if self.thresh is not None:
            adjacency[adjacency > self.thresh] = 0

        edge_list = torch.tensor(np.array(np.nonzero(adjacency)))
        return edge_list


class RAGGraphBuilder(BaseGraphBuilder):
    """
    Region Adjacency Graph builder class.

    Args:
        kernel_size (int, optional): Size of the kernel to detect connectivity. Defaults to 5.
        hops (int, optional): Number of hops in a multi-hop neighbourhood. Defaults to 1.

    Returns:
        A pathml.graph.utils.Graph object containing node and edge information.

    References:
        [1] https://github.com/BiomedSciAI/histocartography/tree/main
        [2] Jaume, G., Pati, P., Anklin, V., Foncubierta, A. and Gabrani, M., 2021, September.
        Histocartography: A toolkit for graph analytics in digital pathology.
        In MICCAI Workshop on Computational Pathology (pp. 117-128). PMLR.
    """

    def __init__(self, kernel_size=3, hops=1, **kwargs):
        """Create a graph builder that uses a provided kernel size to detect connectivity"""
        assert hops > 0 and isinstance(
            hops, int
        ), f"Invalid hops {hops} ({type(hops)}). Must be integer >= 0"
        self.kernel_size = kernel_size
        self.hops = hops
        super().__init__(**kwargs)

    def _set_node_labels(self, annotation, num_nodes):
        """Set the node labels of the graphs using annotation"""
        assert (
            annotation.shape[0] == num_nodes
        ), "Number of annotations do not match number of nodes"
        return annotation

    def _build_topology(self, instance_map):
        """Create the graph topology from the instance connectivty in the instance_map"""

        if instance_map is None:  # pragma: no cover
            raise ValueError("Instance map cannot be None for RAG Graph Building")

        regions = regionprops(instance_map)
        instance_ids = torch.empty(len(regions), dtype=torch.uint8)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        adjacency = np.zeros(shape=(len(instance_ids), len(instance_ids)))

        for instance_id in np.arange(1, len(instance_ids) + 1):
            mask = (instance_map == instance_id).astype(np.uint8)
            dilation = cv2.dilate(mask, kernel, iterations=1)
            boundary = dilation - mask
            idx = pd.unique(instance_map[boundary.astype(bool)])
            instance_id -= 1  # because instance_map id starts from 1
            idx -= 1  # because instance_map id starts from 1
            adjacency[instance_id, idx] = 1

        edge_list = torch.tensor(np.array(np.nonzero(adjacency)))

        for _ in range(self.hops - 1):
            edge_list = two_hop(edge_list, self.num_nodes)
        return edge_list


class MSTGraphBuilder(BaseGraphBuilder):
    """
    Minimum Spanning Tree Graph class for graph building.

    Args:
        k (int, optional): Number of neighbors. Defaults to 5.
        thresh (int, optional): Maximum allowed distance between 2 nodes. Defaults to None (no thresholding).

    Returns:
        A pathml.graph.utils.Graph object containing node and edge information.
    """

    def __init__(self, k=5, thresh=None, **kwargs):
        """Create a graph builder that uses the (thresholded) kNN algorithm to define the graph topology."""

        self.k = k
        self.thresh = thresh
        super().__init__(**kwargs)

    def _set_node_labels(self, annotation, num_nodes):
        """Set the node labels of the graphs using annotation"""
        assert (
            annotation.shape[0] == num_nodes
        ), "Number of annotations do not match number of nodes"
        return annotation

    def _build_topology(self, annotation):
        """Build topology using (thresholded) kNN"""

        # build kNN adjacency
        adjacency = kneighbors_graph(
            self.centroids,
            self.k,
            mode="distance",
            include_self=False,
            metric="euclidean",
        ).toarray()

        # filter edges that are too far (ie larger than thresh)
        if self.thresh is not None:
            adjacency[adjacency > self.thresh] = 0

        adjacency_nz = adjacency.nonzero()
        num_edges = np.array(adjacency_nz).T.shape[0]
        edges_and_weights = np.hstack(
            [
                np.transpose(adjacency_nz),
                np.reshape(adjacency[adjacency_nz], (num_edges, 1)),
            ]
        )
        knn_graph = nx.Graph()
        for i, j, weight in edges_and_weights:
            knn_graph.add_edge(i, j, weight=weight)

        mst_graph = nx.minimum_spanning_tree(knn_graph, weight="weight")
        graph = from_networkx(mst_graph)
        edge_list = graph.edge_index
        return edge_list


class SuperpixelExtractor:
    """Helper class to extract superpixels from images

    Args:
       nr_superpixels (None, int): The number of super pixels before any merging.
       superpixel_size (None, int): The size of super pixels before any merging.
       max_nr_superpixels (int, optional): Upper bound for the number of super pixels.
                                           Useful when providing a superpixel size.
       blur_kernel_size (float, optional): Size of the blur kernel. Defaults to 0.
       compactness (int, optional): Compactness of the superpixels. Defaults to 30.
       max_iterations (int, optional): Number of iterations of the slic algorithm. Defaults to 10.
       threshold (float, optional): Connectivity threshold. Defaults to 0.03.
       connectivity (int, optional): Connectivity for merging graph. Defaults to 2.
       downsampling_factor (int, optional): Downsampling factor from the input image
                                            resolution. Defaults to 1.

    References:
        [1] https://github.com/BiomedSciAI/histocartography/tree/main
        [2] Jaume, G., Pati, P., Anklin, V., Foncubierta, A. and Gabrani, M., 2021, September.
        Histocartography: A toolkit for graph analytics in digital pathology.
        In MICCAI Workshop on Computational Pathology (pp. 117-128). PMLR.
    """

    def __init__(
        self,
        nr_superpixels: int = None,
        superpixel_size: int = None,
        max_nr_superpixels=None,
        blur_kernel_size=1,
        compactness=20,
        max_iterations=10,
        threshold=0.03,
        connectivity=2,
        color_space="rgb",
        downsampling_factor=1,
        **kwargs,
    ):
        """Abstract class that extracts superpixels from RGB Images"""

        assert (nr_superpixels is None and superpixel_size is not None) or (
            nr_superpixels is not None and superpixel_size is None
        ), "Provide value for either nr_superpixels or superpixel_size"
        self.nr_superpixels = nr_superpixels
        self.superpixel_size = superpixel_size
        self.max_nr_superpixels = max_nr_superpixels
        self.blur_kernel_size = blur_kernel_size
        self.compactness = compactness
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.connectivity = connectivity
        self.color_space = color_space
        self.downsampling_factor = downsampling_factor
        super().__init__(**kwargs)

    def process(self, input_image, tissue_mask=None):  # type: ignore[override]
        """Return the superpixels of a given input image"""
        original_height, original_width, _ = input_image.shape
        if self.downsampling_factor != 1:
            input_image = self._downsample(input_image, self.downsampling_factor)
            if tissue_mask is not None:
                tissue_mask = self._downsample(tissue_mask, self.downsampling_factor)
        superpixels = self._extract_superpixels(
            image=input_image, tissue_mask=tissue_mask
        )
        if self.downsampling_factor != 1:
            superpixels = self._upsample(superpixels, original_height, original_width)
        return superpixels

    @abstractmethod
    def _extract_superpixels(self, image, tissue_mask=None):
        """Perform the superpixel extraction"""

    @staticmethod
    def _downsample(image, downsampling_factor):
        """Downsample an input image with a given downsampling factor"""
        height, width = image.shape[0], image.shape[1]
        new_height = math.floor(height / downsampling_factor)
        new_width = math.floor(width / downsampling_factor)
        downsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return downsampled_image

    @staticmethod
    def _upsample(image, new_height, new_width):
        """Upsample an input image to a speficied new height and width"""
        upsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return upsampled_image


class SLICSuperpixelExtractor(SuperpixelExtractor):
    """Use the SLIC algorithm to extract superpixels.

    References:
        [1] https://github.com/BiomedSciAI/histocartography/tree/main
        [2] Jaume, G., Pati, P., Anklin, V., Foncubierta, A. and Gabrani, M., 2021, September.
        Histocartography: A toolkit for graph analytics in digital pathology.
        In MICCAI Workshop on Computational Pathology (pp. 117-128). PMLR.
    """

    def __init__(self, **kwargs):
        """Extract superpixels with the SLIC algorithm"""
        super().__init__(**kwargs)

    def _get_nr_superpixels(self, image):
        """Compute the number of superpixels for initial segmentation"""
        if self.superpixel_size is not None:
            nr_superpixels = int(
                (image.shape[0] * image.shape[1] / self.superpixel_size)
            )
        elif self.nr_superpixels is not None:
            nr_superpixels = self.nr_superpixels
        if self.max_nr_superpixels is not None:
            nr_superpixels = min(nr_superpixels, self.max_nr_superpixels)
        return nr_superpixels

    def _extract_superpixels(self, image, *args, **kwargs):
        """Perform the superpixel extraction"""
        if self.color_space == "hed":
            image = rgb2hed(image)
        nr_superpixels = self._get_nr_superpixels(image)

        slic_args = {
            "image": image,
            "sigma": self.blur_kernel_size,
            "n_segments": nr_superpixels,
            "compactness": self.compactness,
            "start_label": 1,
        }
        if skimage.__version__ < "0.20.0":  # pragma: no cover
            slic_args["max_iter"] = self.max_iterations
        else:
            slic_args["max_num_iter"] = self.max_iterations

        superpixels = slic(**slic_args)
        return superpixels


class MergedSuperpixelExtractor(SuperpixelExtractor):
    """Use the SLIC algorithm to extract superpixels and a merging function to merge superpixels

    References:
        [1] https://github.com/BiomedSciAI/histocartography/tree/main
        [2] Jaume, G., Pati, P., Anklin, V., Foncubierta, A. and Gabrani, M., 2021, September.
        Histocartography: A toolkit for graph analytics in digital pathology.
        In MICCAI Workshop on Computational Pathology (pp. 117-128). PMLR.

    """

    def __init__(self, **kwargs):
        """Extract superpixels with the SLIC algorithm and then merge"""
        super().__init__(**kwargs)

    def _get_nr_superpixels(self, image):
        """Compute the number of superpixels for initial segmentation"""
        if self.superpixel_size is not None:
            nr_superpixels = int(
                (image.shape[0] * image.shape[1] / self.superpixel_size)
            )
        elif self.nr_superpixels is not None:
            nr_superpixels = self.nr_superpixels
        if self.max_nr_superpixels is not None:
            nr_superpixels = min(nr_superpixels, self.max_nr_superpixels)
        return nr_superpixels

    def _extract_initial_superpixels(self, image):
        """Extract initial superpixels using SLIC"""
        nr_superpixels = self._get_nr_superpixels(image)

        slic_args = {
            "image": image,
            "sigma": self.blur_kernel_size,
            "n_segments": nr_superpixels,
            "compactness": self.compactness,
            "start_label": 1,
        }
        if skimage.__version__ < "0.20.0":
            slic_args["max_iter"] = self.max_iterations
        else:
            slic_args["max_num_iter"] = self.max_iterations

        superpixels = slic(**slic_args)
        return superpixels

    def _merge_superpixels(self, input_image, initial_superpixels, tissue_mask=None):
        """Merge the initial superpixels to return merged superpixels"""
        if tissue_mask is not None:  # pragma: no cover
            # Remove superpixels belonging to background or having < 10% tissue
            # content
            ids_initial = np.unique(initial_superpixels, return_counts=True)
            ids_masked = np.unique(
                tissue_mask * initial_superpixels, return_counts=True
            )

            ctr = 1
            superpixels = np.zeros_like(initial_superpixels)
            for i in range(len(ids_initial[0])):
                id = ids_initial[0][i]
                if id in ids_masked[0]:
                    idx = np.where(id == ids_masked[0])[0]
                    ratio = ids_masked[1][idx] / ids_initial[1][i]
                    if ratio >= 0.1:
                        superpixels[initial_superpixels == id] = ctr
                        ctr += 1

            initial_superpixels = superpixels

        # Merge superpixels within tissue region
        g = self._generate_graph(input_image, initial_superpixels)

        merged_superpixels = graph.merge_hierarchical(
            initial_superpixels,
            g,
            thresh=self.threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=self._merging_function,
            weight_func=self._weighting_function,
        )
        merged_superpixels += 1  # Handle regionprops that ignores all values of 0
        mask = np.zeros_like(initial_superpixels)
        mask[initial_superpixels != 0] = 1
        merged_superpixels = merged_superpixels * mask
        return merged_superpixels

    @abstractmethod
    def _generate_graph(self, input_image, superpixels):
        """Generate a graph based on the input image and initial superpixel segmentation."""

    @abstractmethod
    def _weighting_function(self, graph, src, dst, n):
        """Handle merging of nodes of a region boundary region adjacency graph."""

    @abstractmethod
    def _merging_function(self, graph, src, dst):
        """Call back called before merging 2 nodes."""

    def _extract_superpixels(self, image, tissue_mask=None):
        """Perform superpixel extraction"""
        initial_superpixels = self._extract_initial_superpixels(image)
        merged_superpixels = self._merge_superpixels(
            image, initial_superpixels, tissue_mask
        )

        return merged_superpixels, initial_superpixels

    def process(self, input_image, tissue_mask=None):  # type: ignore[override]
        """Return the superpixels of a given input image"""
        original_height, original_width, _ = input_image.shape
        if self.downsampling_factor is not None and self.downsampling_factor != 1:
            input_image = self._downsample(input_image, self.downsampling_factor)
            if tissue_mask is not None:
                tissue_mask = self._downsample(tissue_mask, self.downsampling_factor)
        merged_superpixels, initial_superpixels = self._extract_superpixels(
            input_image, tissue_mask
        )
        if self.downsampling_factor != 1:
            merged_superpixels = self._upsample(
                merged_superpixels, original_height, original_width
            )
            initial_superpixels = self._upsample(
                initial_superpixels, original_height, original_width
            )
        return merged_superpixels, initial_superpixels


class ColorMergedSuperpixelExtractor(MergedSuperpixelExtractor):
    """Superpixel merger based on color attibutes taken from the HACT-Net Implementation
    Args:
        w_hist (float, optional): Weight of the histogram features for merging. Defaults to 0.5.
        w_mean (float, optional): Weight of the mean features for merging. Defaults to 0.5.

    References:
        [1] https://github.com/BiomedSciAI/histocartography/tree/main
        [2] Jaume, G., Pati, P., Anklin, V., Foncubierta, A. and Gabrani, M., 2021, September.
        Histocartography: A toolkit for graph analytics in digital pathology.
        In MICCAI Workshop on Computational Pathology (pp. 117-128). PMLR.
    """

    def __init__(self, w_hist: float = 0.5, w_mean: float = 0.5, **kwargs):
        self.w_hist = w_hist
        self.w_mean = w_mean
        super().__init__(**kwargs)

    def _color_features_per_channel(self, img_ch: np.ndarray) -> np.ndarray:
        """Extract color histograms from image channel"""
        hist, _ = np.histogram(img_ch, bins=np.arange(0, 257, 64))  # 8 bins
        return hist

    def _generate_graph(self, input_image, superpixels):
        """Construct RAG graph using initial superpixel instance map"""
        g = graph.RAG(superpixels, connectivity=self.connectivity)
        if 0 in g.nodes:
            g.remove_node(n=0)  # remove background node

        for n in g:
            g.nodes[n].update(
                {
                    "labels": [n],
                    "N": 0,
                    "x": np.array([0, 0, 0]),
                    "y": np.array([0, 0, 0]),
                    "r": np.array([]),
                    "g": np.array([]),
                    "b": np.array([]),
                }
            )

        for index in np.ndindex(superpixels.shape):
            current = superpixels[index]
            if current == 0:
                continue
            g.nodes[current]["N"] += 1
            g.nodes[current]["x"] += input_image[index]
            g.nodes[current]["y"] = np.vstack(
                (g.nodes[current]["y"], input_image[index])
            )

        for n in g:
            g.nodes[n]["mean"] = g.nodes[n]["x"] / g.nodes[n]["N"]
            g.nodes[n]["mean"] = g.nodes[n]["mean"] / np.linalg.norm(g.nodes[n]["mean"])

            g.nodes[n]["y"] = np.delete(g.nodes[n]["y"], 0, axis=0)
            g.nodes[n]["r"] = self._color_features_per_channel(g.nodes[n]["y"][:, 0])
            g.nodes[n]["g"] = self._color_features_per_channel(g.nodes[n]["y"][:, 1])
            g.nodes[n]["b"] = self._color_features_per_channel(g.nodes[n]["y"][:, 2])

            g.nodes[n]["r"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["r"])
            g.nodes[n]["g"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["g"])
            g.nodes[n]["b"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["b"])

        for x, y, d in g.edges(data=True):
            diff_mean = np.linalg.norm(g.nodes[x]["mean"] - g.nodes[y]["mean"]) / 2

            diff_r = np.linalg.norm(g.nodes[x]["r"] - g.nodes[y]["r"]) / 2
            diff_g = np.linalg.norm(g.nodes[x]["g"] - g.nodes[y]["g"]) / 2
            diff_b = np.linalg.norm(g.nodes[x]["b"] - g.nodes[y]["b"]) / 2
            diff_hist = (diff_r + diff_g + diff_b) / 3

            diff = self.w_hist * diff_hist + self.w_mean * diff_mean

            d["weight"] = diff

        return g

    def _weighting_function(self, graph, src, dst, n):
        diff_mean = np.linalg.norm(graph.nodes[dst]["mean"] - graph.nodes[n]["mean"])

        diff_r = np.linalg.norm(graph.nodes[dst]["r"] - graph.nodes[n]["r"]) / 2
        diff_g = np.linalg.norm(graph.nodes[dst]["g"] - graph.nodes[n]["g"]) / 2
        diff_b = np.linalg.norm(graph.nodes[dst]["b"] - graph.nodes[n]["b"]) / 2
        diff_hist = (diff_r + diff_g + diff_b) / 3

        diff = self.w_hist * diff_hist + self.w_mean * diff_mean

        return {"weight": diff}

    def _merging_function(self, graph, src, dst):
        graph.nodes[dst]["x"] += graph.nodes[src]["x"]
        graph.nodes[dst]["N"] += graph.nodes[src]["N"]
        graph.nodes[dst]["mean"] = graph.nodes[dst]["x"] / graph.nodes[dst]["N"]
        graph.nodes[dst]["mean"] = graph.nodes[dst]["mean"] / np.linalg.norm(
            graph.nodes[dst]["mean"]
        )

        graph.nodes[dst]["y"] = np.vstack(
            (graph.nodes[dst]["y"], graph.nodes[src]["y"])
        )
        graph.nodes[dst]["r"] = self._color_features_per_channel(
            graph.nodes[dst]["y"][:, 0]
        )
        graph.nodes[dst]["g"] = self._color_features_per_channel(
            graph.nodes[dst]["y"][:, 1]
        )
        graph.nodes[dst]["b"] = self._color_features_per_channel(
            graph.nodes[dst]["y"][:, 2]
        )

        graph.nodes[dst]["r"] = graph.nodes[dst]["r"] / np.linalg.norm(
            graph.nodes[dst]["r"]
        )
        graph.nodes[dst]["g"] = graph.nodes[dst]["r"] / np.linalg.norm(
            graph.nodes[dst]["g"]
        )
        graph.nodes[dst]["b"] = graph.nodes[dst]["r"] / np.linalg.norm(
            graph.nodes[dst]["b"]
        )
