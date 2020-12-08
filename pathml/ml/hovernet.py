import torch
from torch import nn

## HoVer-Net
## See Fig. 2 in Graham et al. 2019 paper

class _BatchNormRelu(nn.Module):
    """BatchNorm + Relu layer"""
    def __init__(self, n_channels):
        super(_BatchNormRelu, self).__init__()
        self.batch_norm = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        return self.relu(self.batch_norm(inputs))


class _HoVerNetResidualUnit(nn.Module):
    """
    Residual unit.
    See: Fig. 2(a) from Graham et al. 2019 HoVer-Net paper.
    This unit is not preactivated! That's handled when assembling units into blocks.
    output_channels corresponds to m in the figure
    """
    def __init__(self, input_channels, output_channels, stride):
        super(_HoVerNetResidualUnit, self).__init__()
        internal_channels = output_channels // 4
        if stride != 1 or input_channels != output_channels:
            self.convshortcut = nn.Conv2d(input_channels, output_channels, kernel_size = 1,
                                          stride = stride, padding = 0, dilation = 1, bias = False)
        else:
            self.convshortcut = None
        self.conv1 = nn.Conv2d(input_channels, internal_channels, kernel_size = 1, bias = False)
        self.bnrelu1 = _BatchNormRelu(internal_channels)
        self.conv2 = nn.Conv2d(input_channels, internal_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bnrelu2 = _BatchNormRelu(internal_channels)
        self.conv3 = nn.Conv2d(internal_channels, output_channels, kernel_size = 1, bias = False)

    def forward(self, inputs):
        skip = self.convshortcut(inputs) if self.convshortcut else inputs
        out = self.conv1(inputs)
        out = self.bnrelu1(out)
        out = self.conv2(out)
        out = self.bnrelu2(out)
        out = out + skip
        return out


def _make_HoVerNet_residual_block(input_channels, output_channels, stride, n_units):
    """
    Stack multiple residual units into a block.
    output_channels is given as m in Fig. 2 from Graham et al. 2019 paper
    """
    units = []
    # first unit in block is different
    units.append(_HoVerNetResidualUnit(input_channels, output_channels, stride))

    for i in range(n_units-1):
        units.append(_HoVerNetResidualUnit(output_channels, output_channels, stride = 1))
        # add a final activation ('preact' for the next unit)
        # This is different from how authors implemented - they added BNRelu before all units except the first, plus
        # a final one at the end.
        # I think this is equivalent to just adding a BNRelu after each unit
        units.append(_BatchNormRelu(output_channels))

    return nn.Sequential(*units)


class _HoVerNetEncoder(nn.Module):
    """
    Encoder for HoVer-Net.
    7x7 conv, then four residual blocks, then 1x1 conv.
    BatchNormRelu after first convolution, based on code from authors, see:
     (https://github.com/vqdang/hover_net/blob/5d1560315a3de8e7d4c8122b97b1fe9b9513910b/src/model/graph.py#L67)
    """
    def __init__(self):
        super(_HoVerNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, padding = 3)
        self.bnrelu1 = _BatchNormRelu(64)
        self.block1 = _make_HoVerNet_residual_block(input_channels=64, output_channels = 256, stride = 1, n_units = 3)
        self.block2 = _make_HoVerNet_residual_block(input_channels=256, output_channels = 512, stride = 1, n_units = 4)
        self.block3 = _make_HoVerNet_residual_block(input_channels=512, output_channels = 1024, stride = 1, n_units = 6)
        self.block4 = _make_HoVerNet_residual_block(input_channels=1024, output_channels = 2048, stride = 1, n_units= 3)
        self.conv2 = nn.Conv2d(in_channels = 2048, out_channels = 1024, kernel_size = 1, padding = 0)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bnrelu1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.conv2(out)
        return out


def center_crop_im_batch(batch, dims, batch_order = "BHWC"):
    """
    Center crop images in a batch.

    Args:
        batch: The batch of images to be cropped
        dims: Amount to be cropped (tuple for H, W)
    """
    assert batch.ndim == 4, f"ERROR input shape is {batch.shape} - expecting a batch with 4 dimensions total"

    crop_t = dims[0] // 2
    crop_b = dims[0] - crop_t
    crop_l = dims[1] // 2
    crop_r = dims[1] - crop_l

    if batch_order == "BHWC":
        assert batch.shape[3] == 3
        batch_cropped = batch[:, crop_t:-crop_b, crop_l:-crop_r, :]
    elif batch_order == "BCHW":
        assert batch.shape[1] == 3
        batch_cropped = batch[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        raise Exception("Input batch order not valid")

    return batch_cropped


class _HoVerNetDenseUnit(nn.Module):
    """
    Dense unit.
    See: Fig. 2(b) from Graham et al. 2019 HoVer-Net paper.
    """
    def __init__(self, input_channels):
        super(_HoVerNetDenseUnit, self).__init__()
        self.bnrelu1 = _BatchNormRelu(input_channels)
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = 128, kernel_size = 1)
        self.bnrelu2 = _BatchNormRelu(128)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 5)

    def forward(self, inputs):
        out = self.bnrelu1(inputs)
        out = self.conv1(out)
        out = self.bnrelu2(out)
        out = self.conv2(out)

        # need to make sure that inputs have same shape as out, so that we can concat
        cropdims = (inputs.size(2) - out.size(2), inputs.size(3) - out.size(3))
        if cropdims == (0, 0):
            # no cropping necessary
            inputs_cropped = inputs
        else:
            inputs_cropped = center_crop_im_batch(inputs, dims = cropdims)

        out = torch.cat((inputs_cropped, out), dim = 1)
        return out


def _make_HoVerNet_dense_block(input_channels, n_units):
    """
    Stack multiple dense units into a block.
    """
    units = []
    in_dim = input_channels
    for i in range(n_units):
        units.append(_HoVerNetDenseUnit(in_dim))
        in_dim += 32
    units.append(_BatchNormRelu(in_dim))
    return nn.Sequential(*units)


class _HoverNetDecoder(nn.Module):
    """
    One of the three identical decoder branches.
    """
    def __init__(self):
        super(_HoverNetDecoder, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 5, stride = 1, bias = False)
        self.dense1 = _make_HoVerNet_dense_block(input_channels = 256, n_units = 8)
        self.conv2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 1, stride = 1, bias = False)
        self.upsample2 = nn.Upsample(scale_factor = 2)
        self.conv3 = nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 5, stride = 1, bias = False)
        self.dense2 = _make_HoVerNet_dense_block(input_channels = 128, n_units = 4)

        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 1, stride = 1, bias = False)
        self.upsample3 = nn.Upsample(scale_factor = 2)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 5,
                               stride = 1, bias = False, padding = 2)

    def forward(self, inputs):
        out = self.upsample1(inputs)
        out = self.conv1(out)
        out = self.dense1(out)
        out = self.conv2(out)
        out = self.upsample2(out)
        out = self.conv3(out)
        out = self.dense2(out)
        out = self.conv4(out)
        out = self.upsample3(out)
        out = self.conv5(out)
        return out


class HoverNet(nn.Module):
    """
    HoVer-Net.

    References
        https://arxiv.org/pdf/1812.06499.pdf
    """
    def __init__(self):
        self.encoder = _HoVerNetEncoder()

        self.np_branch = _HoverNetDecoder()     # np = nuclear pixel
        self.hv_branch = _HoverNetDecoder()     # hv = horizontal vertival
        self.nc_branch = _HoverNetDecoder()     # nc = nuclear classification

        # TODO: implement heads to put on top of the branches

    def forward(self, inputs):
        encoded = self.encoder(inputs)

        out_np = self.np_branch(encoded)
        out_hv = self.hv_branch(encoded)
        out_nc = self.nc_branch(encoded)

        return [out_np, out_hv, out_nc]


