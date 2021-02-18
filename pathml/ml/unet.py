import torch
from torch import nn
from torchvision.transforms import CenterCrop
from torch.nn.functional import interpolate


class _UNetConvBlock(nn.Module):
    """
    Convolution block for U-Net

    From the paper:
    The contracting path follows the typical architecture of a convolutional network.
    It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed
    by a rectified linear unit (ReLU)...
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = out_c, out_channels = out_c, kernel_size = 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class _UNetUpConvBlock(nn.Module):
    """
    Up-Convolution block for U-Net

    From the paper:
    Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution
    (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly
    cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels = in_c, out_channels = out_c, kernel_size = 2, stride = 2)
        self.conv = _UNetConvBlock(in_c = in_c, out_c = out_c)

    def forward(self, x, x_skip):
        """
        x is the input
        x_skip is the skip connection from the encoder block
        """
        x = self.up(x)
        # crop tensor from skip connection to match H and W of x
        x_skip = CenterCrop((x.shape[2], x.shape[3]))(x_skip)
        x = torch.cat([x, x_skip], dim = 1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net is a convolutional network for biomedical image segmentation.
    The architecture consists of a contracting path to capture context and a symmetric expanding
    path that enables precise localization.

    As described in the original paper, by default no padding is used, so the dimensions get smaller each layer.
    Input of size 572px will lead to output of size 388px (See Fig. 1 in the paper).
    The ``keep_dim`` parameter can be used to enfore the output to be the same shape as the input.

    Code is based on:
        https://amaarora.github.io/2020/09/13/unet.html
        https://github.com/LeeJunHyun/Image_Segmentation

    Args:
        in_channels (int): Number of channels in input. E.g. 3 for RGB image
        out_channels (int): Number of channels in output. E.g. 1 for a binary classification setting.
        keep_dim (bool): Whether to enforce output to match the dimensions of input. If ``True``, a final interpolation
            step will be applied. Defaults to ``False``.

    References:
        Ronneberger, O., Fischer, P. and Brox, T., 2015, October.
        U-net: Convolutional networks for biomedical image segmentation.
        In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241).
        Springer, Cham.
    """

    def __init__(self, in_channels=3, out_channels=1, keep_dim=False):
        super().__init__()
        self.keep_dim = keep_dim
        self.pool = nn.MaxPool2d(2)

        self.conv1 = _UNetConvBlock(in_c = in_channels, out_c = 64)
        self.conv2 = _UNetConvBlock(in_c = 64, out_c = 128)
        self.conv3 = _UNetConvBlock(in_c = 128, out_c = 256)
        self.conv4 = _UNetConvBlock(in_c = 256, out_c = 512)
        self.conv5 = _UNetConvBlock(in_c = 512, out_c = 1024)

        self.upconv1 = _UNetUpConvBlock(in_c = 1024, out_c = 512)
        self.upconv2 = _UNetUpConvBlock(in_c = 512, out_c = 256)
        self.upconv3 = _UNetUpConvBlock(in_c = 256, out_c = 128)
        self.upconv4 = _UNetUpConvBlock(in_c = 128, out_c = 64)

        self.head = nn.Conv2d(in_channels = 64, out_channels = out_channels, kernel_size = 1)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        x5 = self.pool(x4)
        x5 = self.conv5(x5)

        # decoder
        up1 = self.upconv1(x = x5, x_skip = x4)
        up2 = self.upconv2(x = up1, x_skip = x3)
        up3 = self.upconv3(x = up2, x_skip = x2)
        up4 = self.upconv4(x = up3, x_skip = x1)
        out = self.head(up4)

        if self.keep_dim:
            out = interpolate(out, size = (x.shape[2], x.shape[3]))

        return out
