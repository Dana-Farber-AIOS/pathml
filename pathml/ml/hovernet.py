import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

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
        print(out1.shape)
        out1 = self.bnrelu1(out1)
        print(out1.shape)

        out1 = self.block1(out1)
        print(out1.shape)

        out2 = self.block2(out1)
        print(out2.shape)

        out3 = self.block3(out2)
        print(out3.shape)

        out4 = self.block4(out3)
        print(out4.shape)
        out4 = self.conv2(out4)
        print(out4.shape)

        return [out1, out2, out3, out4]


def center_crop_im_batch(batch, dims, batch_order = "BCHW"):
    """
    Center crop images in a batch.

    Args:
        batch: The batch of images to be cropped
        dims: Amount to be cropped (tuple for H, W)
    """
    assert batch.ndim == 4, f"ERROR input shape is {batch.shape} - expecting a batch with 4 dimensions total"
    assert len(dims) == 2, f"ERROR input cropping dims is {dims} - expecting a tuple with 2 elements total"
    assert batch_order in {"BHCW", "BCHW"}, \
        f"ERROR input batch order {batch_order} not recognized. Must be one of 'BHCW' or 'BCHW'"

    if dims == (0, 0):
        # no cropping necessary in this case
        batch_cropped = batch
    else:
        crop_t = dims[0] // 2
        crop_b = dims[0] - crop_t
        crop_l = dims[1] // 2
        crop_r = dims[1] - crop_l

        if batch_order == "BHWC":
            batch_cropped = batch[:, crop_t:-crop_b, crop_l:-crop_r, :]
        elif batch_order == "BCHW":
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
        print(out.shape)
        # skip connection addition
        out = out + block3_out
        print(out.shape)
        out = self.conv1(out)
        print(out.shape)
        out = self.dense1(out)
        print(out.shape)
        out = self.conv2(out)
        print(out.shape)
        out = self.upsample2(out)
        print(out.shape)
        # skip connection
        out = out + block2_out
        print(out.shape)
        out = self.conv3(out)
        print(out.shape)
        out = self.dense2(out)
        print(out.shape)
        out = self.conv4(out)
        print(out.shape)
        out = self.upsample3(out)
        print(out.shape)
        # last skip connection
        out = out + block1_out
        print(out.shape)
        out = self.conv5(out)
        print(out.shape)
        return out


class HoverNet(nn.Module):
    """
    HoVer-Net.
    Each branch returns logits.

    Args:
        n_classes (int): Number of classes for classification task. If ``None`` then the classification branch is not
            used.

    References
        https://arxiv.org/pdf/1812.06499.pdf
    """
    def __init__(self, n_classes=None):
        super().__init__()
        self.n_classes = n_classes
        self.encoder = _HoVerNetEncoder()

        # NP branch (nuclear pixel)
        self.np_branch = _HoverNetDecoder()
        # classification head
        self.np_head = nn.Sequential(
            _BatchNormRelu(n_channels = 64),
            # two channels in output - background prob and pixel prob
            nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1)
        )

        # HV branch (horizontal vertical)
        self.hv_branch = _HoverNetDecoder()  # hv = horizontal vertical
        # classification head
        self.hv_head = nn.Sequential(
            _BatchNormRelu(n_channels = 64),
            # two channels in output - horizontal and vertical
            nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1)
        )

        # NC branch (nuclear classification)
        # If n_classes is none, then we are in nucleus detection, not classification, so we don't use this branch
        if self.n_classes is not None:
            self.nc_branch = _HoverNetDecoder()
            # classification head
            self.nc_head = nn.Sequential(
                _BatchNormRelu(n_channels = 64),
                # one channel in output for each class
                nn.Conv2d(in_channels = 64, out_channels = self.n_classes, kernel_size = 1)
            )

    def forward(self, inputs):
        encoded = self.encoder(inputs)

        for i, block_output in enumerate(encoded):
            print(f"block {i} output shape: {block_output.shape}")

        print("NP branch starting:")
        out_np = self.np_branch(encoded)
        out_np = self.np_head(out_np)

        print("HV branch starting:")
        out_hv = self.hv_branch(encoded)
        out_hv = self.hv_head(out_hv)

        outputs = [out_np, out_hv]

        if self.n_classes is not None:
            print("NC branch starting:")
            out_nc = self.nc_branch(encoded)
            out_nc = self.nc_head(out_nc)
            outputs.append(out_nc)

        return outputs


## loss functions and associated utils


def dice_loss(true, logits, eps=1e-3):
    """
    Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return 1 - dice loss.
    From: https://github.com/kevinzakka/pytorch-goodies/blob/c039691f349be9f21527bb38b907a940bfc5e8f3/losses.py#L54

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    assert true.dtype == torch.long, f"Input 'true' is of type {true.type}. It should be a long."
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    loss = (2. * intersection / (cardinality + eps)).mean()
    loss = 1 - loss
    return loss


def convert_multiclass_mask_to_binary(mask):
    """
    Input mask of shape (B, n_classes, H, W) is converted to a mask of shape (B, 1, H, W).
    The last channel is assumed to be background, so the binary mask is computed by taking its inverse.
    """
    m = torch.tensor(1) - mask[:, -1, :, :]
    m = m.unsqueeze(dim = 1)
    return m


def dice_loss_np_head(np_out, true_mask, epsilon=1e-3):
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

    true_mask = convert_multiclass_mask_to_binary(true_mask)
    true_mask = true_mask.type(torch.long)
    loss = dice_loss(logits = preds, true = true_mask, eps = epsilon)
    return loss


def dice_loss_nc_head(nc_out, true_mask, epsilon=1e-3):
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


def ce_loss_nc_head(nc_out, true_mask):
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


def ce_loss_np_head(np_out, true_mask):
    """
    Cross-entropy loss term for nc branch.
    Args:
        np_out: logit outputs of np branch. Tensor of shape (B, 2, H, W)
        true_mask: True mask. Tensor of shape (B, n_classes, H, W)
    """
    truth = convert_multiclass_mask_to_binary(true_mask).type(torch.long).squeeze(dim = 1)
    ce = nn.CrossEntropyLoss()
    loss = ce(np_out, truth)
    return loss

import cv2

def compute_hv_map(mask):
    """
    Preprocessing step for HoVer-Net architecture.
    Compute center of mass for each nucleus, then compute distance of each nuclear pixel to its corresponding center
    of mass.
    Nuclear pixel distances are normalized to (-1, 1). Background pixels are left as 0.
    Operates on a single mask -- should be used at Dataset object.

    Returns an array of shape (2, H, W), where the first channel corresponds to horizontal and second vertical.
    Based on https://github.com/vqdang/hover_net/blob/195ed9b6cc67b12f908285492796fb5c6c15a000/src/loader/augs.py#L192

    Args:
        mask (np.ndarray): Mask indicating individual nuclei. Array of shape (n_classes, H, W),
            where each pixel is in {0, ..., n} with 0 indicating background pixels and {1, ..., n} indicating
            n unique nuclei.
    """
    out = np.zeros((2, mask.shape[1], mask.shape[2]))
    # each individual nucleus is indexed with a different number
    inst_list = list(np.unique(mask))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        # get the mask for the nucleus
        inst_map = mask == inst_id
        inst_map = inst_map.squeeze(axis = 0).astype(np.uint8)
        contours, _ = cv2.findContours(inst_map, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_NONE)

        # assert len(contours) == 1, f"error - found more than one contour! (inst_id = {inst_id})"

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


def get_sobel_kernels(size, dt=torch.float32):
    """
    Create horizontal and vertical Sobel kernels for approximating gradients
    Returned kernels will be of shape (size, size)
    """
    assert size % 2 == 1, "Size must be odd"

    h_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype = torch.float32)
    v_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype = torch.float32)
    h, v = torch.meshgrid([h_range, v_range])
    h, v = h.transpose(0, 1), v.transpose(0, 1)

    kernel_h = h / (h * h + v * v + 1e-5)
    kernel_v = v / (h * h + v * v + 1e-5)

    kernel_h = kernel_h.type(dt)
    kernel_v = kernel_v.type(dt)

    return kernel_h, kernel_v


def get_gradient_hv(hv_batch, kernel_size=5):
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
    """
    assert hv_batch.shape[1] == 2, f"inputs have shape {hv_batch.shape}. Expecting tensor of shape (B, 2, H, W)"
    h_kernel, v_kernel = get_sobel_kernels(kernel_size, dt = hv_batch.dtype)
    # add extra dims so we can convolve with a batch
    h_kernel = h_kernel.unsqueeze(0).unsqueeze(0)
    v_kernel = v_kernel.unsqueeze(0).unsqueeze(0)

    # get the inputs for the h and v channels
    h_inputs = hv_batch[:, 0, :, :].unsqueeze(dim = 1)
    v_inputs = hv_batch[:, 1, :, :].unsqueeze(dim = 1)

    h_grad = F.conv2d(h_inputs, h_kernel, stride = 1, padding = 2)
    v_grad = F.conv2d(v_inputs, v_kernel, stride = 1, padding = 2)

    return h_grad, v_grad


def loss_hv_grad(hv_out, true_hv, nucleus_pixel_mask):
    """
    Equation 3 from HoVer-Net paper for calculating loss for HV predictions.
    Mask is used to compute the hv loss ONLY for nuclear pixels

    Args:
        hv_out: Ouput of hv branch. Tensor of shape (B, 2, H, W)
        true_hv: Ground truth hv maps. Tensor of shape (B, 2, H, W)
        nucleus_pixel_mask: Boolean mask indicating nuclear pixels. Tensor of shape (B, H, W)
    """
    pred_grad_h, pred_grad_v = get_gradient_hv(hv_out)
    true_grad_h, true_grad_v = get_gradient_hv(true_hv)

    # pull out only the values from nuclear pixels
    pred_h = torch.masked_select(pred_grad_h, mask = nucleus_pixel_mask)
    true_h = torch.masked_select(true_grad_h, mask = nucleus_pixel_mask)
    pred_v = torch.masked_select(pred_grad_v, mask = nucleus_pixel_mask)
    true_v = torch.masked_select(true_grad_v, mask = nucleus_pixel_mask)

    loss_h = F.mse_loss(pred_h, true_h)
    loss_v = F.mse_loss(pred_v, true_v)

    loss = loss_h + loss_v
    return loss


def loss_hv_mse(hv_out, true_hv):
    """
    Equation 2 from HoVer-Net paper for calculating loss for HV predictions.

    Args:
        hv_out: Ouput of hv branch. Tensor of shape (B, 2, H, W)
        true_hv: Ground truth hv maps. Tensor of shape (B, 2, H, W)
    """
    loss = F.mse_loss(hv_out, true_hv)
    return loss


def loss_HoVerNet(outputs, ground_truth, n_classes=None):
    """
    Compute loss for HoVer-Net.

    Args:
        outputs: Output of HoVer-Net. Should be a list of [np, hv] if n_classes is None, or a list of [np, hv, nc] if
            n_classes is not None.
            Shapes of each should be:

                - np: (B, 2, H, W)
                - hv: (B, 2, H, W)
                - nc: (B, n_classes, H, W)

        ground_truth: True labels. Should be a list of [mask, hv], where mask is of shape (B, 1, H, W) if n_classes is
            None or (B, n_classes, H, W) if n_classes is not None.
            hv is a tensor of precomputed horizontal and vertical distances
            of nuclear pixels to their corresponding centers of mass, and is of shape (B, 2, H, W).
        n_classes (int): Number of classes for classification task. If ``None`` then the classification branch is not
            used.
    """
    true_mask, true_hv = ground_truth
    true_hv = true_hv.float()
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
    np_loss_dice = dice_loss_np_head(np_out, true_mask)
    np_loss_ce = ce_loss_np_head(np_out, true_mask)

    hv_loss_grad = loss_hv_grad(hv, true_hv, nucleus_mask)
    hv_loss_mse = loss_hv_mse(hv, true_hv)

    # authors suggest using coefficient of 2 for hv gradient loss term
    hv_loss_grad = 2*hv_loss_grad

    if n_classes is not None:
        nc_loss_dice = dice_loss_nc_head(nc, true_mask)
        nc_loss_ce = ce_loss_nc_head(nc, true_mask)
    else:
        nc_loss_dice = 0
        nc_loss_ce = 0

    loss = np_loss_dice + np_loss_ce + hv_loss_mse + hv_loss_grad + nc_loss_dice + nc_loss_ce
    return loss

