import os
import tempfile

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

# Assuming TileDataset is in pathml.ml, adjust the import as necessary
from pathml.datasets.datasets import TileDataset
from pathml.graph.utils import HACTPairData


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


def fake_hactnet_inputs():
    """fake batch of input for HACTNet"""
    cell_features = torch.rand(200, 256)
    cell_edge_index = erdos_renyi_graph(200, 0.2, directed=False)
    tissue_features = torch.rand(20, 256)
    tissue_edge_index = erdos_renyi_graph(20, 0.2, directed=False)
    target = torch.tensor([1, 2])
    assignment = torch.randint(low=0, high=20, size=(200,)).long()
    data = HACTPairData(
        x_cell=cell_features,
        edge_index_cell=cell_edge_index,
        x_tissue=tissue_features,
        edge_index_tissue=tissue_edge_index,
        assignment=assignment,
        target=target,
    )
    return data
