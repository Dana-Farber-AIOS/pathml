import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2
from skimage.segmentation import watershed
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from warnings import warn

from pathml.preprocessing.utils import segmentation_lines
from pathml.ml.utils import center_crop_im_batch, dice_loss, get_sobel_kernels


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
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bnrelu2 = _BatchNormRelu(internal_channels)
        self.conv3 = nn.Conv2d(internal_channels, output_channels, kernel_size = 1, bias = False)

    def forward(self, inputs):
        skip = self.convshortcut(inputs) if self.convshortcut else inputs
        out = self.conv1(inputs)
        out = self.bnrelu1(out)
        out = self.conv2(out)
        out = self.bnrelu2(out)
        out = self.conv3(out)
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

     Reuturn a list of the outputs from each residual block, for later skip connections
    """
    def __init__(self):
        super(_HoVerNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, padding = 3)
        self.bnrelu1 = _BatchNormRelu(64)
        self.block1 = _make_HoVerNet_residual_block(input_channels=64, output_channels = 256, stride = 1, n_units = 3)
        self.block2 = _make_HoVerNet_residual_block(input_channels=256, output_channels = 512, stride = 2, n_units = 4)
        self.block3 = _make_HoVerNet_residual_block(input_channels=512, output_channels = 1024, stride = 2, n_units = 6)
        self.block4 = _make_HoVerNet_residual_block(input_channels=1024, output_channels = 2048, stride = 2, n_units= 3)
        self.conv2 = nn.Conv2d(in_channels = 2048, out_channels = 1024, kernel_size = 1, padding = 0)

    def forward(self, inputs):
        out1 = self.conv1(inputs)
        out1 = self.bnrelu1(out1)
        out1 = self.block1(out1)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out4 = self.conv2(out4)
        return [out1, out2, out3, out4]


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
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 5, padding = 2)

    def forward(self, inputs):
        out = self.bnrelu1(inputs)
        out = self.conv1(out)
        out = self.bnrelu2(out)
        out = self.conv2(out)

        # need to make sure that inputs have same shape as out, so that we can concat
        cropdims = (inputs.size(2) - out.size(2), inputs.size(3) - out.size(3))
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
        self.conv1 = nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 5, padding = 2,
                               stride = 1, bias = False)
        self.dense1 = _make_HoVerNet_dense_block(input_channels = 256, n_units = 8)
        self.conv2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 1, stride = 1, bias = False)
        self.upsample2 = nn.Upsample(scale_factor = 2)
        self.conv3 = nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 5, padding = 2,
                               stride = 1, bias = False)
        self.dense2 = _make_HoVerNet_dense_block(input_channels = 128, n_units = 4)

        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 1, stride = 1, bias = False)
        self.upsample3 = nn.Upsample(scale_factor = 2)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 5,
                               stride = 1, bias = False, padding = 2)

    def forward(self, inputs):
        """
        Inputs should be a list of the outputs from each residual block, so that we can use skip connections
        """
        block1_out, block2_out, block3_out, block4_out = inputs
        out = self.upsample1(block4_out)
        # skip connection addition
        out = out + block3_out
        out = self.conv1(out)
        out = self.dense1(out)
        out = self.conv2(out)
        out = self.upsample2(out)
        # skip connection
        out = out + block2_out
        out = self.conv3(out)
        out = self.dense2(out)
        out = self.conv4(out)
        out = self.upsample3(out)
        # last skip connection
        out = out + block1_out
        out = self.conv5(out)
        return out


class HoVerNet(nn.Module):
    """
    Model for simultaneous segmentation and classification based on HoVer-Net.
    Can also be used for segmentation only, if class labels are not supplied.
    Each branch returns logits.

    Args:
        n_classes (int): Number of classes for classification task. If ``None`` then the classification branch is not
            used.

    References:
        Graham, S., Vu, Q.D., Raza, S.E.A., Azam, A., Tsang, Y.W., Kwak, J.T. and Rajpoot, N., 2019.
        Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images.
        Medical Image Analysis, 58, p.101563.
    """
    def __init__(self, n_classes=None):
        super().__init__()
        self.n_classes = n_classes
        self.encoder = _HoVerNetEncoder()

        # NP branch (nuclear pixel)
        self.np_branch = _HoverNetDecoder()
        # classification head
        self.np_head = nn.Sequential(
            # two channels in output - background prob and pixel prob
            nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1)
        )

        # HV branch (horizontal vertical)
        self.hv_branch = _HoverNetDecoder()  # hv = horizontal vertical
        # classification head
        self.hv_head = nn.Sequential(
            # two channels in output - horizontal and vertical
            nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1)
        )

        # NC branch (nuclear classification)
        # If n_classes is none, then we are in nucleus detection, not classification, so we don't use this branch
        if self.n_classes is not None:
            self.nc_branch = _HoverNetDecoder()
            # classification head
            self.nc_head = nn.Sequential(
                # one channel in output for each class
                nn.Conv2d(in_channels = 64, out_channels = self.n_classes, kernel_size = 1)
            )

    def forward(self, inputs):
        encoded = self.encoder(inputs)

        """for i, block_output in enumerate(encoded):
            print(f"block {i} output shape: {block_output.shape}")"""

        out_np = self.np_branch(encoded)
        out_np = self.np_head(out_np)

        out_hv = self.hv_branch(encoded)
        out_hv = self.hv_head(out_hv)

        outputs = [out_np, out_hv]

        if self.n_classes is not None:
            out_nc = self.nc_branch(encoded)
            out_nc = self.nc_head(out_nc)
            outputs.append(out_nc)

        return outputs


# loss functions and associated utils

def _convert_multiclass_mask_to_binary(mask):
    """
    Input mask of shape (B, n_classes, H, W) is converted to a mask of shape (B, 1, H, W).
    The last channel is assumed to be background, so the binary mask is computed by taking its inverse.
    """
    m = torch.tensor(1) - mask[:, -1, :, :]
    m = m.unsqueeze(dim = 1)
    return m


def _dice_loss_np_head(np_out, true_mask, epsilon=1e-3):
    """
    Dice loss term for nuclear pixel branch.
    This will compute dice loss for the entire batch
    (not the same as computing dice loss for each image and then averaging!)

    Args:
        np_out: logit outputs of np branch. Tensor of shape (B, 2, H, W)
        true_mask: True mask. Tensor of shape (B, n_classes, H, W)
        epsilon (float): Epsilon passed to ``dice_loss()``
    """
    # get logits for only the channel corresponding to prediction of 1
    # unsqueeze to keep the dimensions the same
    preds = np_out[:, 1, :, :].unsqueeze(dim = 1)

    true_mask = _convert_multiclass_mask_to_binary(true_mask)
    true_mask = true_mask.type(torch.long)
    loss = dice_loss(logits = preds, true = true_mask, eps = epsilon)
    return loss


def _dice_loss_nc_head(nc_out, true_mask, epsilon=1e-3):
    """
    Dice loss term for nuclear classification branch.
    Computes dice loss for each channel, and sums up.
    This will compute dice loss for the entire batch
    (not the same as computing dice loss for each image and then averaging!)

    Args:
        nc_out: logit outputs of nc branch. Tensor of shape (B, n_classes, H, W)
        true_mask: True mask. Tensor of shape (B, n_classes, H, W)
        epsilon (float): Epsilon passed to ``dice_loss()``
    """
    truth = torch.argmax(true_mask, dim = 1, keepdim=True).type(torch.long)
    loss = dice_loss(logits = nc_out, true = truth, eps = epsilon)
    return loss


def _ce_loss_nc_head(nc_out, true_mask):
    """
    Cross-entropy loss term for nc branch.
    Args:
        nc_out: logit outputs of nc branch. Tensor of shape (B, n_classes, H, W)
        true_mask: True mask. Tensor of shape (B, n_classes, H, W)
    """
    truth = torch.argmax(true_mask, dim = 1).type(torch.long)
    ce = nn.CrossEntropyLoss()
    loss = ce(nc_out, truth)
    return loss


def _ce_loss_np_head(np_out, true_mask):
    """
    Cross-entropy loss term for np branch.
    Args:
        np_out: logit outputs of np branch. Tensor of shape (B, 2, H, W)
        true_mask: True mask. Tensor of shape (B, n_classes, H, W)
    """
    truth = _convert_multiclass_mask_to_binary(true_mask).type(torch.long).squeeze(dim = 1)
    ce = nn.CrossEntropyLoss()
    loss = ce(np_out, truth)
    return loss


def compute_hv_map(mask):
    """
    Preprocessing step for HoVer-Net architecture.
    Compute center of mass for each nucleus, then compute distance of each nuclear pixel to its corresponding center
    of mass.
    Nuclear pixel distances are normalized to (-1, 1). Background pixels are left as 0.
    Operates on a single mask.
    Can be used in Dataset object to make Dataloader compatible with HoVer-Net.

    Based on https://github.com/vqdang/hover_net/blob/195ed9b6cc67b12f908285492796fb5c6c15a000/src/loader/augs.py#L192

    Args:
        mask (np.ndarray): Mask indicating individual nuclei. Array of shape (H, W),
            where each pixel is in {0, ..., n} with 0 indicating background pixels and {1, ..., n} indicating
            n unique nuclei.

    Returns:
        np.ndarray: array of hv maps of shape (2, H, W). First channel corresponds to horizontal and second vertical.
    """
    assert mask.ndim == 2, f"Input mask has shape {mask.shape}. Expecting a mask with 2 dimensions (H, W)"

    out = np.zeros((2, mask.shape[0], mask.shape[1]))
    # each individual nucleus is indexed with a different number
    inst_list = list(np.unique(mask))

    try:
        inst_list.remove(0)  # 0 is background
    except:
        warn("No pixels with 0 label. This means that there are no background pixels."
             "This may indicate a problem. Ignore this warning if this is expected/intended.")

    for inst_id in inst_list:
        # get the mask for the nucleus
        inst_map = mask == inst_id
        inst_map = inst_map.astype(np.uint8)
        contours, _ = cv2.findContours(inst_map, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_NONE)

        # get center of mass coords
        mom = cv2.moments(contours[0])
        com_x = mom["m10"] / (mom["m00"] + 1e-6)
        com_y = mom["m01"] / (mom["m00"] + 1e-6)
        inst_com = (int(com_y), int(com_x))

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype('float32')
        inst_y = inst_y.astype('float32')

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= (-np.amin(inst_x[inst_x < 0]))
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= (-np.amin(inst_y[inst_y < 0]))
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= (np.amax(inst_x[inst_x > 0]))
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= (np.amax(inst_y[inst_y > 0]))

        # add to output mask
        # this works assuming background is 0, and each pixel is assigned to only one nucleus.
        out[0, :, :] += inst_x
        out[1, :, :] += inst_y
    return out


def _get_gradient_hv(hv_batch, kernel_size=5):
    """
    Calculate the horizontal partial differentiation for horizontal channel
    and the vertical partial differentiation for vertical channel.
    The partial differentiation is approximated by calculating the central differnce
    which is obtained by using Sobel kernel of size 5x5. The boundary is zero-padded
    when channel is convolved with the Sobel kernel.

    Args:
        hv_batch: tensor of shape (B, 2, H, W). Channel index 0 for horizonal maps and 1 for vertical maps.
            These maps are distance from each nuclear pixel to center of mass of corresponding nucleus.
        kernel_size (int): width of kernel to use for gradient approximation.

    Returns:
        Tuple of (h_grad, v_grad) where each is a Tensor giving horizontal and vertical gradients respectively
    """
    assert hv_batch.shape[1] == 2, f"inputs have shape {hv_batch.shape}. Expecting tensor of shape (B, 2, H, W)"
    h_kernel, v_kernel = get_sobel_kernels(kernel_size, dt = hv_batch.dtype)
    
    # move kernels to same device as batch
    h_kernel = h_kernel.to(hv_batch.device)
    v_kernel = v_kernel.to(hv_batch.device)
    
    # add extra dims so we can convolve with a batch
    h_kernel = h_kernel.unsqueeze(0).unsqueeze(0)
    v_kernel = v_kernel.unsqueeze(0).unsqueeze(0)

    # get the inputs for the h and v channels
    h_inputs = hv_batch[:, 0, :, :].unsqueeze(dim = 1)
    v_inputs = hv_batch[:, 1, :, :].unsqueeze(dim = 1)

    h_grad = F.conv2d(h_inputs, h_kernel, stride = 1, padding = 2)
    v_grad = F.conv2d(v_inputs, v_kernel, stride = 1, padding = 2)
    
    del h_kernel
    del v_kernel
    
    return h_grad, v_grad


def _loss_hv_grad(hv_out, true_hv, nucleus_pixel_mask):
    """
    Equation 3 from HoVer-Net paper for calculating loss for HV predictions.
    Mask is used to compute the hv loss ONLY for nuclear pixels

    Args:
        hv_out: Ouput of hv branch. Tensor of shape (B, 2, H, W)
        true_hv: Ground truth hv maps. Tensor of shape (B, 2, H, W)
        nucleus_pixel_mask: Boolean mask indicating nuclear pixels. Tensor of shape (B, H, W)
    """
    pred_grad_h, pred_grad_v = _get_gradient_hv(hv_out)
    true_grad_h, true_grad_v = _get_gradient_hv(true_hv)

    # pull out only the values from nuclear pixels
    pred_h = torch.masked_select(pred_grad_h, mask = nucleus_pixel_mask)
    true_h = torch.masked_select(true_grad_h, mask = nucleus_pixel_mask)
    pred_v = torch.masked_select(pred_grad_v, mask = nucleus_pixel_mask)
    true_v = torch.masked_select(true_grad_v, mask = nucleus_pixel_mask)

    loss_h = F.mse_loss(pred_h, true_h)
    loss_v = F.mse_loss(pred_v, true_v)

    loss = loss_h + loss_v
    return loss


def _loss_hv_mse(hv_out, true_hv):
    """
    Equation 2 from HoVer-Net paper for calculating loss for HV predictions.

    Args:
        hv_out: Ouput of hv branch. Tensor of shape (B, 2, H, W)
        true_hv: Ground truth hv maps. Tensor of shape (B, 2, H, W)
    """
    loss = F.mse_loss(hv_out, true_hv)
    return loss


def loss_hovernet(outputs, ground_truth, n_classes=None):
    """
    Compute loss for HoVer-Net.
    Equation (1) in Graham et al.

    Args:
        outputs: Output of HoVer-Net. Should be a list of [np, hv] if n_classes is None, or a list of [np, hv, nc] if
            n_classes is not None.
            Shapes of each should be:

                - np: (B, 2, H, W)
                - hv: (B, 2, H, W)
                - nc: (B, n_classes, H, W)

        ground_truth: True labels. Should be a list of [mask, hv], where mask is a Tensor of shape (B, 1, H, W)
            if n_classes is ``None`` or (B, n_classes, H, W) if n_classes is not ``None``.
            hv is a tensor of precomputed horizontal and vertical distances
            of nuclear pixels to their corresponding centers of mass, and is of shape (B, 2, H, W).
        n_classes (int): Number of classes for classification task. If ``None`` then the classification branch is not
            used.

    References:
        Graham, S., Vu, Q.D., Raza, S.E.A., Azam, A., Tsang, Y.W., Kwak, J.T. and Rajpoot, N., 2019.
        Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images.
        Medical Image Analysis, 58, p.101563.
    """
    true_mask, true_hv = ground_truth
    # unpack outputs, and also calculate nucleus masks
    if n_classes is None:
        np_out, hv = outputs
        nucleus_mask = true_mask[:, 0, :, :] == 1
    else:
        np_out, hv, nc = outputs
        # in multiclass setting, last channel of masks indicates background, so
        # invert that to get a nucleus mask (Based on convention from PanNuke dataset)
        nucleus_mask = true_mask[:, -1, :, :] == 0

    # from Eq. 1 in HoVer-Net paper, loss function is composed of two terms for each branch.
    np_loss_dice = _dice_loss_np_head(np_out, true_mask)
    np_loss_ce = _ce_loss_np_head(np_out, true_mask)

    hv_loss_grad = _loss_hv_grad(hv, true_hv, nucleus_mask)
    hv_loss_mse = _loss_hv_mse(hv, true_hv)

    # authors suggest using coefficient of 2 for hv gradient loss term
    hv_loss_grad = 2*hv_loss_grad

    if n_classes is not None:
        nc_loss_dice = _dice_loss_nc_head(nc, true_mask)
        nc_loss_ce = _ce_loss_nc_head(nc, true_mask)
    else:
        nc_loss_dice = 0
        nc_loss_ce = 0

    loss = np_loss_dice + np_loss_ce + hv_loss_mse + hv_loss_grad + nc_loss_dice + nc_loss_ce
    return loss


# Post-processing of HoVer-Net outputs

def remove_small_objs(array_in, min_size):
    """
    Removes small foreground regions from binary array, leaving only the contiguous regions which are above
    the size threshold. Pixels in regions below the size threshold are zeroed out.

    Args:
        array_in (np.ndarray): Input array. Must be binary array with dtype=np.uint8.
        min_size (int): Minimum size of each region.

    Returns:
        np.ndarray: Array of labels for regions above the threshold. Each separate contiguous region is labelled with
            a different integer from 1 to n, where n is the number of total distinct contiguous regions
    """
    assert array_in.dtype == np.uint8, f"Input dtype is {array_in.dtype}. Must be np.uint8"
    # remove elements below size threshold
    # each contiguous nucleus region gets a unique id
    n_labels, labels = cv2.connectedComponents(array_in)
    # each integer is a different nucleus, so bincount gives nucleus sizes
    sizes = np.bincount(labels.flatten())
    for nucleus_ix, size_ix in zip(range(n_labels), sizes):
        if size_ix < min_size:
            # below size threshold - set all to zero
            labels[labels == nucleus_ix] = 0
    return labels


def _post_process_single_hovernet(np_out, hv_out, small_obj_size_thresh=10, kernel_size=21, h=0.5, k=0.5):
    """
    Combine predictions of np channel and hv channel to create final predictions.
    Works by creating energy landscape from gradients, and the applying watershed segmentation.
    This function works on a single image and is wrapped in ``post_process_batch_hovernet()`` to apply across a batch.
    See: Section B of HoVer-Net article and
    https://github.com/vqdang/hover_net/blob/14c5996fa61ede4691e87905775e8f4243da6a62/models/hovernet/post_proc.py#L27

    Args:
        np_out (torch.Tensor): Output of NP branch. Tensor of shape (2, H, W) of logit predictions for binary classification
        hv_out (torch.Tensor): Output of HV branch. Tensor of shape (2, H, W) of predictions for horizontal/vertical maps
        small_obj_size_thresh (int): Minimum number of pixels in regions. Defaults to 10.
        kernel_size (int): Width of Sobel kernel used to compute horizontal and vertical gradients.
        h (float): hyperparameter for thresholding nucleus probabilities. Defaults to 0.5.
        k (float): hyperparameter for thresholding energy landscape to create markers for watershed
            segmentation. Defaults to 0.5.
    """
    # compute pixel probabilities from logits, apply threshold, and get into np array
    np_preds = F.softmax(np_out, dim = 0)[1, :, :]
    np_preds = np_preds.numpy()

    np_preds[np_preds >= h] = 1
    np_preds[np_preds < h] = 0
    np_preds = np_preds.astype(np.uint8)

    np_preds = remove_small_objs(np_preds, min_size = small_obj_size_thresh)
    # Back to binary. now np_preds corresponds to tau(q, h) from HoVer-Net paper
    np_preds[np_preds > 0] = 1
    tau_q_h = np_preds

    # normalize hv predictions, and compute horizontal and vertical gradients, and normalize again
    hv_out = hv_out.numpy().astype(np.float32)
    h_out = hv_out[0, ...]
    v_out = hv_out[1, ...]
    # https://stackoverflow.com/a/39037135
    h_normed = cv2.normalize(h_out, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    v_normed = cv2.normalize(v_out, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    h_grad = cv2.Sobel(h_normed, cv2.CV_64F, dx = 1, dy = 0, ksize = kernel_size)
    v_grad = cv2.Sobel(v_normed, cv2.CV_64F, dx = 0, dy = 1, ksize = kernel_size)

    h_grad = cv2.normalize(h_grad, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    v_grad = cv2.normalize(v_grad, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    # flip the gradient direction so that highest values are steepest gradient
    h_grad = 1 - h_grad
    v_grad = 1 - v_grad

    S_m = np.maximum(h_grad, v_grad)
    S_m[tau_q_h == 0] = 0
    # energy landscape
    # note that the paper says that they use E = (1 - tau(S_m, k)) * tau(q, h)
    # but in the authors' code the actually use: E = (1 - S_m) * tau(q, h)
    # this actually makes more sense because no need to threshold the energy surface
    energy = (1.0 - S_m) * tau_q_h

    # get markers
    # In the paper it says they use M = sigma(tau(q, h) - tau(S_m, k))
    # But it makes more sense to threshold the energy landscape to get the peaks of hills.
    # Also, the fact they used sigma in the paper makes me think that this is what they intended,
    m = np.array(energy >= k, dtype = np.uint8)
    m = binary_fill_holes(m).astype(np.uint8)
    m = remove_small_objs(m, min_size = small_obj_size_thresh)

    # nuclei values form mountains so inverse to get basins for watershed
    energy = -cv2.GaussianBlur(energy, (3, 3), 0)
    out = watershed(image = energy, markers = m, mask = tau_q_h)

    return out


def post_process_batch_hovernet(outputs, n_classes, small_obj_size_thresh=10, kernel_size=21, h=0.5, k=0.5):
    """
    Post-process HoVer-Net outputs to get a final predicted mask.
    See: Section B of HoVer-Net article and
    https://github.com/vqdang/hover_net/blob/14c5996fa61ede4691e87905775e8f4243da6a62/models/hovernet/post_proc.py#L27

    Args:
        outputs (list): Outputs of HoVer-Net model. List of [np_out, hv_out], or [np_out, hv_out, nc_out]
            depending on whether model is predicting classification or not.

            - np_out is a Tensor of shape (B, 2, H, W) of logit predictions for binary classification
            - hv_out is a Tensor of shape (B, 2, H, W) of predictions for horizontal/vertical maps
            - nc_out is a Tensor of shape (B, n_classes, H, W) of logits for classification

        n_classes (int): Number of classes for classification task. If ``None`` then only segmentation is performed.
        small_obj_size_thresh (int): Minimum number of pixels in regions. Defaults to 10.
        kernel_size (int): Width of Sobel kernel used to compute horizontal and vertical gradients.
        h (float): hyperparameter for thresholding nucleus probabilities. Defaults to 0.5.
        k (float): hyperparameter for thresholding energy landscape to create markers for watershed
            segmentation. Defaults to 0.5.

    Returns:
        np.ndarray: If n_classes is None, returns det_out. In classification setting, returns (det_out, class_out).

            - det_out is np.ndarray of shape (B, H, W)
            - class_out is np.ndarray of shape (B, n_classes, H, W)

            Each pixel is labelled from 0 to n, where n is the number of individual nuclei detected. 0 pixels indicate
            background. Pixel values i indicate that the pixel belongs to the ith nucleus.
    """

    assert len(outputs) in {2, 3}, f"outputs has size {len(outputs)}. Must have size 2 (for segmentation) or 3 (for " \
                                   f"classification)"
    if n_classes is None:
        np_out, hv_out = outputs
        # send ouputs to cpu
        np_out = np_out.detach().cpu()
        hv_out = hv_out.detach().cpu()
        classification = False
    else:
        assert len(outputs) == 3, f"n_classes={n_classes} but outputs has {len(outputs)} elements. Expecting a list " \
                                  f"of length 3, one for each of np, hv, and nc branches"
        np_out, hv_out, nc_out = outputs
        # send ouputs to cpu as np arrays
        np_out = np_out.detach().cpu()
        hv_out = hv_out.detach().cpu()
        nc_out = nc_out.detach().cpu()
        classification = True

    batchsize = hv_out.shape[0]
    # first get the nucleus detection preds
    out_detection_list = []
    for i in range(batchsize):
        preds = _post_process_single_hovernet(np_out[i, ...], hv_out[i, ...], small_obj_size_thresh, kernel_size, h, k)
        out_detection_list.append(preds)
    out_detection = np.stack(out_detection_list)

    if classification:
        # need to do last step of majority vote
        # get the pixel-level class predictions from the logits
        nc_out_preds = F.softmax(nc_out, dim = 1).argmax(dim = 1)

        out_classification = np.zeros_like(nc_out.numpy(), dtype = np.uint8)

        for batch_ix, nuc_preds in enumerate(out_detection_list):
            # get labels of nuclei from nucleus detection
            nucleus_labels = list(np.unique(nuc_preds))
            if 0 in nucleus_labels:
                nucleus_labels.remove(0)  # 0 is background
            nucleus_class_preds = nc_out_preds[batch_ix, ...]

            out_class_preds_single = out_classification[batch_ix, ...]

            # for each nucleus, get the class predictions for the pixels and take a vote
            for nucleus_ix in nucleus_labels:
                # get mask for the specific nucleus
                ix_mask = nuc_preds == nucleus_ix
                votes = nucleus_class_preds[ix_mask]
                majority_class = np.argmax(np.bincount(votes))
                out_class_preds_single[majority_class][ix_mask] = nucleus_ix

            out_classification[batch_ix, ...] = out_class_preds_single

        return out_detection, out_classification
    else:
        return out_detection


# plotting hovernet outputs

def _vis_outputs_single(images, preds, n_classes, index=0, ax=None, markersize=5, palette=None):
    """
    Plot the results of HoVer-Net predictions for a single image, overlayed on the original image.

    Args:
        images: Input RGB image batch. Tensor of shape (B, 3, H, W).
        preds: Postprocessed outputs of HoVer-Net. From post_process_batch_hovernet(). Can be either:
            - Tensor of shape (B, H, W), in the context of nucleus detection.
            - Tensor of shape (B, n_classes, H, W), in the context of nucleus classification.
        n_classes (int): Number of classes for classification setting, or None to indicate detection setting.
        index (int): Index of image to plot.
        ax: Matplotlib axes object to plot on. If None, creates a new plot. Defaults to None.
        markersize: Size of markers used to outline nuclei
        palette (list): list of colors to use for plotting. If None, uses matplotlib.colors.TABLEAU_COLORS.
            Defaults to None
    """
    if palette is None:
        palette = list(TABLEAU_COLORS.values())

    if n_classes is not None:
        classification = True
        n_classes = preds.shape[1]
        assert len(palette) >= n_classes, f"len(palette)={len(palette)} < n_classes={n_classes}."
    else:
        classification = False

    assert len(preds.shape) in [3, 4], f"Preds shape is {preds.shape}. Must be (B, H, W) or (B, n_classes, H, W)"

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(images[index, ...].permute(1, 2, 0))

    if classification is False:
        nucleus_labels = list(np.unique(preds[index, ...]))
        nucleus_labels.remove(0)  # background
        # plot each individual nucleus
        for label in nucleus_labels:
            nuclei_mask = preds[index, ...] == label
            x, y = segmentation_lines(nuclei_mask.astype(np.uint8))
            ax.scatter(x, y, color = palette[0], marker = ".", s = markersize)
    else:
        nucleus_labels = list(np.unique(preds[index, ...]))
        nucleus_labels.remove(0)  # background
        # plot each individual nucleus
        for label in nucleus_labels:
            for i in range(n_classes):
                nuclei_mask = preds[index, i, ...] == label
                x, y = segmentation_lines(nuclei_mask.astype(np.uint8))
                ax.scatter(x, y, color = palette[i], marker = ".", s = markersize)
    ax.axis("off")
