import pytest
import torch

from pathml.ml.unet import UNet


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("out_c", [1, 3])
def test_unet_shapes(out_c, keepdim):
    batch_size = 1
    channels_in = 3
    im_size_in = 572

    x = torch.randn(batch_size, channels_in, im_size_in, im_size_in)

    net = UNet(out_channels = out_c, keep_dim = keepdim)
    out = net(x)

    if keepdim:
        assert out.shape == (batch_size, out_c, im_size_in, im_size_in)
    else:

        # compute output size, if keep_dim is false
        nlayers = 4
        im_size_out = im_size_in
        # at each layer in encoder, two conv layers without padding loses 4px total, followed by a downsample by 2
        for _ in range(nlayers):
            im_size_out = (im_size_out - 4) / 2
        # two more conv layers at the bottom layer
        im_size_out = im_size_out - 4
        # at each layer in decoder, upsample by 2 followed by two conv layers without padding for a loss of 4
        for _ in range(nlayers):
            im_size_out = (im_size_out * 2) - 4

        assert out.shape == (batch_size, out_c, im_size_out, im_size_out)