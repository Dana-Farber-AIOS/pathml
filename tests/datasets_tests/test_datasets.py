import os
import shutil
import tempfile

import h5py
import numpy as np
import pytest
import torch

from pathml.datasets.datasets import EntityDataset, TileDataset

# Assuming TileDataset is in pathml.ml, adjust the import as necessary
from pathml.graph import Graph


@pytest.fixture
def create_test_h5_file():
    """
    Fixture to create a temporary h5 file simulating the output of SlideData processing.
    This file will serve as input for testing TileDataset.
    """
    tmp_dir = tempfile.mkdtemp()
    h5_path = os.path.join(tmp_dir, "test_tile_dataset.h5")

    with h5py.File(h5_path, "w") as f:
        tiles = f.create_group("tiles")
        tiles.attrs["tile_shape"] = "(224, 224, 3)"
        for i in range(5):
            tile = tiles.create_group(str(i))
            array_data = np.random.rand(224, 224, 3).astype(
                "float32"
            )  # Ensure data type matches expected torch.Tensor type
            tile.create_dataset("array", data=array_data)
            if i % 2 == 0:  # Add masks to some tiles
                masks = tile.create_group("masks")
                masks.create_dataset(
                    "mask1",
                    data=np.random.randint(2, size=(224, 224)).astype("float32"),
                )
            labels = tile.create_group("labels").attrs
            labels["example_label"] = "label_value"
        fields = f.create_group("fields")
        labels = fields.create_group("labels")
        labels.attrs["slide_label"] = "slide_value"

    yield h5_path
    os.remove(h5_path)
    os.rmdir(tmp_dir)


def test_tile_dataset_initialization(create_test_h5_file):
    h5_path = create_test_h5_file
    dataset = TileDataset(h5_path)

    assert len(dataset) == 5
    assert dataset.tile_shape == (224, 224, 3)
    assert dataset.slide_level_labels["slide_label"] == "slide_value"


def test_tile_dataset_getitem(create_test_h5_file):
    h5_path = create_test_h5_file
    dataset = TileDataset(h5_path)

    for i in range(len(dataset)):
        im, masks, lab_tile, lab_slide = dataset[i]

        # Image tensor shape should match expected (C, H, W) after transpose
        assert im.shape == (
            3,
            224,
            224,
        ), "Image tensor shape should match expected (C, H, W)"
        if masks is not None:
            assert masks.shape[0] > 0 and masks.shape[1:] == (
                224,
                224,
            ), "Mask shape should be (n_masks, H, W)"
        assert "example_label" in lab_tile, "Tile labels should include 'example_label'"
        assert (
            lab_slide["slide_label"] == "slide_value"
        ), "Slide label should match expected value"


def test_tile_dataset_unsupported_shape_explicit_check(create_test_h5_file):
    h5_path = create_test_h5_file
    dataset = TileDataset(h5_path)

    with h5py.File(h5_path, "r+") as f:
        # Create an unsupported shape explicitly
        del f["tiles"]["0"]["array"]
        f["tiles"]["0"].create_dataset(
            "array", data=np.random.rand(10, 10)
        )  # 2D array, unsupported

    try:
        _ = dataset[0]
        assert False, "NotImplementedError was expected but not raised."
    except NotImplementedError:
        pass  # This is the expected outcome


# Additional test cases can be added here to cover more scenarios, such as different image shapes (e.g., 5D images),
# testing with actual mask data, and ensuring that custom collate_fn behavior is as expected.


def test_tile_dataset_with_masks(create_test_h5_file):
    h5_path = create_test_h5_file
    dataset = TileDataset(h5_path)

    # Assuming the first item has masks
    _, masks, _, _ = dataset[0]
    assert masks is not None, "Masks should be present"
    assert masks.shape[0] > 0, "There should be at least one mask"


def fake_graph_inputs():

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_centroids = torch.randn(3, 2)
    node_features = torch.randn(3, 2)
    target = torch.tensor([1])

    graph_obj = Graph(
        edge_index=edge_index,
        node_centroids=node_centroids,
        node_features=node_features,
        target=target,
    )
    assignment = assignment = torch.randint(low=0, high=3, size=(3, 2)).long()

    return graph_obj, graph_obj, assignment


@pytest.fixture
def create_test_graph_file():
    """
    Fixture to create a temporary h5 file simulating the output of SlideData processing.
    This file will serve as input for testing TileDataset.
    """
    graphs_path = tempfile.mkdtemp()
    os.makedirs(os.path.join(graphs_path, "cell_graphs", "train"), exist_ok=True)
    os.makedirs(os.path.join(graphs_path, "tissue_graphs", "train"), exist_ok=True)
    os.makedirs(
        os.path.join(graphs_path, "assignment_matrices", "train"), exist_ok=True
    )

    cell_graph, tissue_graph, assignment = fake_graph_inputs()

    torch.save(
        cell_graph, os.path.join(graphs_path, "cell_graphs", "train", "example.pt")
    )
    torch.save(
        tissue_graph, os.path.join(graphs_path, "tissue_graphs", "train", "example.pt")
    )
    torch.save(
        assignment,
        os.path.join(graphs_path, "assignment_matrices", "train", "example.pt"),
    )

    yield graphs_path
    os.remove(os.path.join(graphs_path, "cell_graphs", "train", "example.pt"))
    os.remove(os.path.join(graphs_path, "tissue_graphs", "train", "example.pt"))
    os.remove(os.path.join(graphs_path, "assignment_matrices", "train", "example.pt"))
    shutil.rmtree(graphs_path)


def test_entity_dataset(create_test_graph_file):

    graphs_path = create_test_graph_file
    train_dataset = EntityDataset(
        os.path.join(graphs_path, "cell_graphs/train/"),
        os.path.join(graphs_path, "tissue_graphs/train/"),
        os.path.join(graphs_path, "assignment_matrices/train/"),
    )
    batch = train_dataset[0]

    assert batch.x_cell.shape == (3, 2)
    assert batch.x_tissue.shape == (3, 2)
    assert batch.edge_index_cell.shape == (2, 4)
    assert batch.edge_index_tissue.shape == (2, 4)
    assert len(train_dataset) == 1
