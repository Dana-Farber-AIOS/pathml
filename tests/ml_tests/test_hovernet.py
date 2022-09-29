"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest
import torch

from pathml.ml import hovernet


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
    l = hovernet.loss_hovernet(outputs, truth, n_classes=n_classes)
    assert l.item() > 0


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
