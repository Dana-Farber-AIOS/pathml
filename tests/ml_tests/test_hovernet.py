"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest
import torch

from pathml.ml.models import hovernet


def fake_hovernet_inputs(n_classes, batch_size=2):
    """fake batch of input for HoVer-Net"""
    if n_classes is None:
        n_classes = 1
    ims = torch.rand(size=(batch_size, 3, 256, 256))
    masks = torch.randint(low=0, high=2, size=(batch_size, n_classes, 256, 256))
    hv = torch.rand(size=(batch_size, 2, 256, 256))
    return ims, masks, hv


def fake_hovernet_output(n_classes, batch_size=2):
    """fake batch of output from HoVer-Net"""
    np_out = torch.rand(size=(batch_size, 2, 256, 256))
    hv_out = torch.rand(size=(batch_size, 2, 256, 256))
    if n_classes:
        nc_out = torch.rand(size=(batch_size, n_classes, 256, 256))
        return np_out, hv_out, nc_out
    else:
        return np_out, hv_out


@pytest.mark.parametrize("n_classes", [None, 2])
def test_hovernet_forward_pass(n_classes):
    """Make sure that dimensions of outputs are as expected from forward pass"""
    batch_size = 1
    ims, _, _ = fake_hovernet_inputs(n_classes, batch_size=batch_size)
    hover_net = hovernet.HoVerNet(n_classes=n_classes)
    with torch.no_grad():
        outputs = hover_net(ims)

    np_out, hv_out = outputs[0], outputs[1]
    assert np_out.shape == (batch_size, 2, 256, 256)
    assert hv_out.shape == (batch_size, 2, 256, 256)

    if n_classes:
        nc_out = outputs[2]
        assert nc_out.shape == (batch_size, n_classes, 256, 256)


@pytest.mark.parametrize("n_classes", [None, 2])
def test_hovernet_loss(n_classes):
    """make sure that loss function produces a loss"""
    ims, masks, hv = fake_hovernet_inputs(n_classes)
    truth = [masks, hv]
    outputs = fake_hovernet_output(n_classes)
    loss = hovernet.loss_hovernet(outputs, truth, n_classes=n_classes)
    assert loss.item() > 0


# TODO test each individual loss function


@pytest.mark.parametrize("n_classes", [None, 4])
def test_postprocess(n_classes):
    batch_size = 2
    outputs = fake_hovernet_output(n_classes=n_classes, batch_size=batch_size)
    post_processed = hovernet.post_process_batch_hovernet(
        outputs, n_classes=n_classes, small_obj_size_thresh=0
    )
    if n_classes is None:
        assert post_processed.shape == (batch_size, 256, 256)
    else:
        det_out, class_out = post_processed
        assert det_out.shape == (batch_size, 256, 256)
        assert class_out.shape == (batch_size, n_classes, 256, 256)

    return_nc_out_preds = True
    if return_nc_out_preds and n_classes:
        post_processed = hovernet.post_process_batch_hovernet(
            outputs,
            n_classes=n_classes,
            small_obj_size_thresh=0,
            return_nc_out_preds=True,
        )

        det_out, class_out, nc_outpreds = post_processed

        assert det_out.shape == (batch_size, 256, 256)
        assert class_out.shape == (batch_size, n_classes, 256, 256)
        assert nc_outpreds.shape == (batch_size, 256, 256)


def test_remove_small_objs():
    a = np.zeros((100, 100), dtype=np.uint8)
    a[40:60, 40:60] = 1
    a_small_obj = a.copy()
    # add a couple small objects
    a_small_obj[81:83, 81:83] = 2
    a_small_obj[91:93, 0:2] = 9
    a_removed = hovernet.remove_small_objs(a_small_obj, min_size=10)
    assert np.array_equal(a, a_removed)


def test_compute_hv_map():
    mask = np.random.randint(low=0, high=2, size=(256, 256))
    mask_hv = hovernet.compute_hv_map(mask)
    assert mask_hv.shape == (2, 256, 256)


@pytest.fixture
def create_mock_data():
    # Create a mock instance mask with two cells
    pred_inst = np.zeros((100, 100), dtype=np.uint8)
    pred_inst[20:40, 20:40] = 1  # Cell 1
    pred_inst[60:80, 60:80] = 2  # Cell 2

    # Create a mock type mask
    pred_type = np.zeros_like(pred_inst)
    pred_type[20:40, 20:40] = 1  # Cell 1 is of type 1
    pred_type[60:80, 60:80] = 2  # Cell 2 is of type 2

    return pred_inst, pred_type


def test_cell_count(create_mock_data):
    pred_inst, pred_type = create_mock_data
    result = hovernet.extract_nuclei_info(pred_inst, pred_type)
    assert len(result) == 2, "Incorrect number of cells detected"


def test_cell_type_and_probability(create_mock_data):
    pred_inst, pred_type = create_mock_data
    result = hovernet.extract_nuclei_info(pred_inst, pred_type)
    assert result[1]["type"] == 1, "Incorrect type for cell 1"
    assert result[2]["type"] == 2, "Incorrect type for cell 2"
    assert np.isclose(
        result[1]["centroid"], [29.5, 29.5]
    ).all(), "Incorrect centroid for cell 1"
    assert np.isclose(
        result[2]["centroid"], [69.5, 69.5]
    ).all(), "Incorrect centroid for cell 2"
    assert np.isclose(result[1]["prob"], 1.0), "Incorrect probability for cell 1"
    assert np.isclose(result[2]["prob"], 1.0), "Incorrect probability for cell 2"


def test_group_centroids_valid_input():
    cell_dict = {
        1: {"centroid": [10, 20], "type": 1, "prob": 0.9},
        2: {"centroid": [30, 40], "type": 2, "prob": 0.8},
        3: {"centroid": [50, 60], "type": 1, "prob": 0.7},
    }
    prob_threshold = 0.75
    grouped = hovernet.group_centroids_by_type(cell_dict, prob_threshold)
    assert 1 in grouped and [10, 20] in grouped[1]
    assert 3 not in grouped  # Cell 2 should be excluded due to low probability
    assert len(grouped[1]) == 1  # Only one cell of type 1 should be included


def test_group_centroids_empty_input():
    cell_dict = {}
    prob_threshold = 0.5
    grouped = hovernet.group_centroids_by_type(cell_dict, prob_threshold)
    assert grouped == {}


def test_group_centroids_invalid_cell_dict():
    cell_dict = "not a dictionary"
    prob_threshold = 0.5
    with pytest.raises(ValueError):
        hovernet.group_centroids_by_type(cell_dict, prob_threshold)


def test_group_centroids_invalid_prob_threshold():
    cell_dict = {1: {"centroid": [10, 20], "type": 1, "prob": 0.9}}
    prob_threshold = -0.1  # Negative threshold
    with pytest.raises(ValueError):
        hovernet.group_centroids_by_type(cell_dict, prob_threshold)
