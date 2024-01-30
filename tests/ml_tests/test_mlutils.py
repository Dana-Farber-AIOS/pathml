from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data

from pathml.ml.utils import (
    broadcast,
    center_crop_im_batch,
    dice_score,
    get_class_weights,
    get_degree_histogram,
    scatter_sum,
    wrap_transform_multichannel,
)


def test_scatter_sum():
    src = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
    index = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    out = scatter_sum(src, index, dim=0, dim_size=2)
    expected = torch.tensor(
        [9, 6], dtype=torch.float
    )  # Sum of values at indices 0 and 1
    assert torch.equal(out, expected), "scatter_sum does not match expected output"


# Test cases
@pytest.mark.parametrize(
    "src,other,dim,expected_shape",
    [
        (torch.tensor([1, 2, 3]), torch.zeros((3, 3)), 1, (3, 3)),
        (torch.tensor([1, 2, 3]), torch.zeros((3, 3, 3)), 1, (3, 3, 3)),
        (torch.tensor([1, 2, 3]), torch.zeros((3, 3)), -1, (3, 3)),
        (torch.tensor([1]), torch.zeros((5, 5, 5)), 0, (5, 5, 5)),
        (torch.tensor([1, 2, 3, 4]), torch.zeros((2, 2, 4)), 2, (2, 2, 4)),
    ],
)
def test_broadcast(src, other, dim, expected_shape):
    result = broadcast(src, other, dim)
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.shape}"
    # Verify that the broadcasted values are correct
    if src.dim() == 1:
        for i in range(src.size(0)):
            assert torch.all(
                result.select(dim, i) == src[i]
            ).item(), "Broadcasted values do not match the source tensor."


def test_get_degree_histogram():
    loader = [
        Data(
            edge_index=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long),
            x=torch.randn(3, 10),
        )
    ]
    edge_index_str, x_str = "edge_index", "x"
    histogram = get_degree_histogram(loader, edge_index_str, x_str)
    expected = torch.tensor([0, 3], dtype=torch.long)  # Each node has a degree of 1
    assert torch.equal(histogram, expected), "Degree histogram does not match expected"


def test_get_class_weights():
    loader = [
        Data(target=torch.tensor([0])),
        Data(target=torch.tensor([1])),
        Data(target=torch.tensor([0])),
    ]
    weights = get_class_weights(loader)
    expected = compute_class_weight(
        "balanced", classes=np.unique([0, 1, 0]), y=np.array([0, 1, 0])
    )
    assert np.allclose(weights, expected), "Class weights do not match expected"


@pytest.mark.parametrize(
    "pred, truth, expected_score",
    [
        (np.array([[1, 1], [0, 0]]), np.array([[1, 1], [0, 0]]), 1.0),
        (np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]]), 0.0),
        (np.array([[1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]), 2 * 1 / (2 + 2)),
        (np.array([[1, 0], [1, 0]]), np.array([[0, 0], [0, 0]]), 0.0),
        (np.array([[2, 0], [5, 0]]), np.array([[0, 0], [0, 0]]), 0.0),
        pytest.param(
            np.array([[1, 0], [1, 0]]),
            np.array([[1]]),
            None,
            marks=pytest.mark.xfail(raises=AssertionError),
        ),
    ],
)
def test_dice_score(pred, truth, expected_score):
    score = dice_score(pred, truth)
    # Allow a small margin of error for floating point arithmetic
    tolerance = 1e-3  # Adjust this tolerance as needed
    assert np.isclose(
        score, expected_score, atol=tolerance
    ), f"Expected score {expected_score}, got {score}"


@pytest.mark.parametrize(
    "input_shape, dims, batch_order, expected_shape",
    [
        ((2, 3, 10, 10), (4, 4), "BCHW", (2, 3, 6, 6)),  # Basic Cropping
        ((2, 3, 10, 10), (0, 0), "BCHW", (2, 3, 10, 10)),  # No Cropping
        ((2, 10, 10, 3), (4, 4), "BHWC", (2, 6, 6, 3)),  # Batch Order BHWC
        ((2, 3, 10, 10), (10, 10), "BCHW", (2, 3, 0, 0)),  # Edge Cropping
    ],
)
def test_center_crop_im_batch(input_shape, dims, batch_order, expected_shape):
    batch = torch.randn(input_shape)
    cropped = center_crop_im_batch(batch, dims, batch_order)
    assert (
        cropped.shape == expected_shape
    ), f"Expected cropped shape {expected_shape}, got {cropped.shape}"


@pytest.mark.parametrize("batch_order", ["BCHW", "BHWC"])  # Valid batch orders
def test_center_crop_im_batch_invalid_dims(batch_order):
    batch = torch.randn((2, 3, 10, 10))
    with pytest.raises(AssertionError):
        center_crop_im_batch(batch, (10, 10, 10), batch_order)  # Invalid dims


@pytest.mark.parametrize("batch_order", ["BCHW", "BHWC"])  # Valid batch orders
def test_center_crop_im_batch_invalid_input_shape(batch_order):
    batch = torch.randn((2, 10, 10))  # Invalid input shape (missing channel dimension)
    with pytest.raises(AssertionError):
        center_crop_im_batch(batch, (4, 4), batch_order)


def test_center_crop_im_batch_invalid_batch_order():
    batch = torch.randn((2, 3, 10, 10))
    with pytest.raises(Exception):
        center_crop_im_batch(batch, (4, 4), "INVALID")  # Invalid batch order


def mock_transform_function(*args, **kwargs):
    # Simulate transforming each mask channel and returning them with the same shape
    transformed_masks = {key: np.copy(kwargs[key]) for key in kwargs if "mask" in key}
    return transformed_masks


def test_wrap_transform_multichannel_single_channel():
    # Simulate an Albumentations transform with a MagicMock
    mock_transform = MagicMock()
    # Pretend the transform modifies the input image/mask in a predictable way (e.g., adds 1)
    mock_transform.side_effect = lambda **kwargs: {
        key: value + 1 for key, value in kwargs.items()
    }
    # Set up additional_targets to mimic an Albumentations transform expecting single-channel input
    mock_transform.additional_targets = {"mask0": "mask"}

    # The wrap_transform_multichannel function should now wrap this behavior
    wrapped_transform = wrap_transform_multichannel(mock_transform)

    # Single channel mask
    mask = np.ones((1, 256, 256), dtype=np.uint8)

    # Apply the wrapped transform
    result = wrapped_transform(mask=mask)

    # Check if the transformed mask matches the expected modification
    expected_output = mask + 1  # Since our mock_transform adds 1
    assert np.array_equal(
        result["mask"], expected_output
    ), "The output mask does not match the expected result."


def test_wrap_transform_multichannel_multi_channel():
    # Simulate an Albumentations transform with a MagicMock
    mock_transform = MagicMock()
    # Pretend the transform modifies the input image/mask in a predictable way (e.g., adds 1 to each channel)
    mock_transform.side_effect = lambda **kwargs: {
        key: value + 1 for key, value in kwargs.items()
    }
    # Set up additional_targets to mimic an Albumentations transform expecting multi-channel input
    mock_transform.additional_targets = {
        "mask0": "mask",
        "mask1": "mask",
        "mask2": "mask",
    }

    # The wrap_transform_multichannel function should now wrap this behavior
    wrapped_transform = wrap_transform_multichannel(mock_transform)

    # Multi-channel mask
    mask = np.random.rand(3, 256, 256).astype(np.uint8)

    # Apply the wrapped transform
    result = wrapped_transform(mask=mask)

    # Check if the transformed mask matches the expected modification
    expected_output = mask + 1  # Since our mock_transform adds 1 to each channel
    assert np.array_equal(
        result["mask"], expected_output
    ), "The output mask does not match the expected result."
