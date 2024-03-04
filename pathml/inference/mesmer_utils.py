"""
The following functions were taken from the Deepcell package to enable PathML to support their Mesmer segmentation model.

Deepcell website:
https://deepcell.readthedocs.io/en/master/#

Citation: 
"Greenwald NF, Miller G, Moen E, Kong A, Kagel A, Dougherty T, Fullaway CC, McIntosh BJ, Leow KX, Schwartz MS, Pavelchek C. Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning. Nature biotechnology. 2022 Apr;40(4):555-65."
"""

import warnings

import cv2
import numpy as np
import scipy.ndimage as nd
from skimage import transform
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import (
    ball,
    cube,
    dilation,
    disk,
    h_maxima,
    remove_small_holes,
    remove_small_objects,
    square,
)
from skimage.segmentation import find_boundaries, relabel_sequential, watershed


def erode_edges(mask, erosion_width):
    """Erode edge of objects to prevent them from touching

    https://github.com/vanvalenlab/deepcell-toolbox/blob/master/deepcell_toolbox/utils.py

    Args:
        mask (numpy.array): uniquely labeled instance mask
        erosion_width (int): integer value for pixel width to erode edges

    Returns:
        numpy.array: mask where each instance has had the edges eroded

    Raises:
        ValueError: mask.ndim is not 2 or 3
    """

    if mask.ndim not in {2, 3}:
        raise ValueError(
            "erode_edges expects arrays of ndim 2 or 3."
            "Got ndim: {}".format(mask.ndim)
        )
    if erosion_width:
        new_mask = np.copy(mask)
        for _ in range(erosion_width):
            boundaries = find_boundaries(new_mask, mode="inner")
            new_mask[boundaries > 0] = 0
        return new_mask

    return mask


def fill_holes(label_img, size=10, connectivity=1):
    """Fills holes located completely within a given label with pixels of the same value

    https://github.com/vanvalenlab/deepcell-toolbox/blob/master/deepcell_toolbox/utils.py

    Args:
        label_img (numpy.array): a 2D labeled image
        size (int): maximum size for a hole to be filled in
        connectivity (int): the connectivity used to define the hole

    Returns:
        numpy.array: a labeled image with no holes smaller than ``size``
            contained within any label.
    """
    output_image = np.copy(label_img)

    props = regionprops(np.squeeze(label_img.astype("int")), cache=False)
    for prop in props:
        if prop.euler_number < 1:

            patch = output_image[prop.slice]

            filled = remove_small_holes(
                ar=(patch == prop.label), area_threshold=size, connectivity=connectivity
            )

            output_image[prop.slice] = np.where(filled, prop.label, patch)

    return output_image


def percentile_threshold(image, percentile=99.9):
    """Threshold an image to reduce bright spots
    This function is from "Greenwald NF, Miller G, Moen E, Kong A, Kagel A, Dougherty T, Fullaway CC, McIntosh BJ, Leow KX, Schwartz MS, Pavelchek C. Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning. Nature biotechnology. 2022 Apr;40(4):555-65."

    https://github.com/vanvalenlab/deepcell-toolbox/blob/master/deepcell_toolbox/processing.py


    Args:
        image: numpy array of image data
        percentile: cutoff used to threshold image

    Returns:
        np.array: thresholded version of input image

    """

    processed_image = np.zeros_like(image)
    for img in range(image.shape[0]):
        for chan in range(image.shape[-1]):
            current_img = np.copy(image[img, ..., chan])
            non_zero_vals = current_img[np.nonzero(current_img)]

            # only threshold if channel isn't blank
            if len(non_zero_vals) > 0:
                img_max = np.percentile(non_zero_vals, percentile)

                # threshold values down to max
                threshold_mask = current_img > img_max
                current_img[threshold_mask] = img_max

                # update image
                processed_image[img, ..., chan] = current_img

    return processed_image


def histogram_normalization(image, kernel_size=None):
    """Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).


    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.

    This function is from "Greenwald NF, Miller G, Moen E, Kong A, Kagel A, Dougherty T, Fullaway CC, McIntosh BJ, Leow KX, Schwartz MS, Pavelchek C. Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning. Nature biotechnology. 2022 Apr;40(4):555-65."

    https://github.com/vanvalenlab/deepcell-toolbox/blob/master/deepcell_toolbox/processing.py

    Args:
        image (numpy.array): numpy array of phase image data.
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.

    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """
    # if not np.issubdtype(image.dtype, np.floating):
    #     logging.info('Converting image dtype to float')
    image = image.astype("float32")

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
            sample_value = X[(0,) * X.ndim]
            if (X == sample_value).all():
                # TODO: Deal with constant value arrays
                # https://github.com/scikit-image/scikit-image/issues/4596
                # logging.warning('Found constant value array in batch %s and '
                #                 'channel %s. Normalizing as zeros.',
                #                 batch, channel)
                image[batch, ..., channel] = np.zeros_like(X)
                continue

            # X = rescale_intensity(X, out_range='float')
            X = rescale_intensity(X, out_range=(0.0, 1.0))
            X = equalize_adapthist(X, kernel_size=kernel_size)
            image[batch, ..., channel] = X
    return image


# pre- and post-processing functions
def mesmer_preprocess(image, **kwargs):
    """Preprocess input data for Mesmer model.

    This function is from "Greenwald NF, Miller G, Moen E, Kong A, Kagel A, Dougherty T, Fullaway CC, McIntosh BJ, Leow KX, Schwartz MS, Pavelchek C. Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning. Nature biotechnology. 2022 Apr;40(4):555-65."

    https://github.com/vanvalenlab/deepcell-tf/blob/master/deepcell/applications/mesmer.py

    Args:
        image: array to be processed

    Returns:
        np.array: processed image array
    """

    if len(image.shape) != 4:
        raise ValueError(f"Image data must be 4D, got image of shape {image.shape}")

    output = np.copy(image)
    threshold = kwargs.get("threshold", True)
    if threshold:
        percentile = kwargs.get("percentile", 99.9)
        output = percentile_threshold(image=output, percentile=percentile)

    normalize = kwargs.get("normalize", True)
    if normalize:
        kernel_size = kwargs.get("kernel_size", 128)
        output = histogram_normalization(image=output, kernel_size=kernel_size)

    return output


def resize(data, shape, data_format="channels_last", labeled_image=False):
    """Resize the data to the given shape.
    Uses openCV to resize the data if the data is a single channel, as it
    is very fast. However, openCV does not support multi-channel resizing,
    so if the data has multiple channels, use skimage.

    This function is from "Greenwald NF, Miller G, Moen E, Kong A, Kagel A, Dougherty T, Fullaway CC, McIntosh BJ, Leow KX, Schwartz MS, Pavelchek C. Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning. Nature biotechnology. 2022 Apr;40(4):555-65."

    https://github.com/vanvalenlab/deepcell-toolbox/blob/master/deepcell_toolbox/utils.py

    Args:
        data (np.array): data to be reshaped. Must have a channel dimension
        shape (tuple): shape of the output data in the form (x,y).
            Batch and channel dimensions are handled automatically and preserved.
        data_format (str): determines the order of the channel axis,
            one of 'channels_first' and 'channels_last'.
        labeled_image (bool): flag to determine how interpolation and floats are handled based
         on whether the data represents raw images or annotations

    Raises:
        ValueError: ndim of data not 3 or 4
        ValueError: Shape for resize can only have length of 2, e.g. (x,y)

    Returns:
        numpy.array: data reshaped to new shape.
    """
    if len(data.shape) not in {3, 4}:
        raise ValueError(
            "Data must have 3 or 4 dimensions, e.g. "
            "[batch, x, y], [x, y, channel] or "
            "[batch, x, y, channel]. Input data only has {} "
            "dimensions.".format(len(data.shape))
        )

    if len(shape) != 2:
        raise ValueError(
            "Shape for resize can only have length of 2, e.g. (x,y)."
            "Input shape has {} dimensions.".format(len(shape))
        )

    original_dtype = data.dtype

    # cv2 resize is faster but does not support multi-channel data
    # If the data is multi-channel, use skimage.transform.resize
    channel_axis = 0 if data_format == "channels_first" else -1
    batch_axis = -1 if data_format == "channels_first" else 0

    # Use skimage for multichannel data
    if data.shape[channel_axis] > 1:
        # Adjust output shape to account for channel axis
        if data_format == "channels_first":
            shape = tuple([data.shape[channel_axis]] + list(shape))
        else:
            shape = tuple(list(shape) + [data.shape[channel_axis]])

        # linear interpolation (order 1) for image data, nearest neighbor (order 0) for labels
        # anti_aliasing introduces spurious labels, include only for image data
        order = 0 if labeled_image else 1
        anti_aliasing = not labeled_image

        def _resize(d):
            return transform.resize(
                d,
                shape,
                mode="constant",
                preserve_range=True,
                order=order,
                anti_aliasing=anti_aliasing,
            )

    # single channel image, resize with cv2
    else:
        shape = tuple(shape)[::-1]  # cv2 expects swapped axes.

        # linear interpolation for image data, nearest neighbor for labels
        # CV2 doesn't support ints for linear interpolation, set to float for image data
        if labeled_image:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR
            data = data.astype("float32")

        def _resize(d):
            return np.expand_dims(
                cv2.resize(np.squeeze(d), shape, interpolation=interpolation),
                axis=channel_axis,
            )

    # Check for batch dimension to loop over
    if len(data.shape) == 4:
        batch = []
        for i in range(data.shape[batch_axis]):
            d = data[i] if batch_axis == 0 else data[..., i]
            batch.append(_resize(d))
        resized = np.stack(batch, axis=batch_axis)
    else:
        resized = _resize(data)

    return resized.astype(original_dtype)


def mesmer_resize_input(image, image_mpp):
    """Checks if there is a difference between image and model resolution
    and resizes if they are different. Otherwise returns the unmodified
    image.

    This function is from "Greenwald NF, Miller G, Moen E, Kong A, Kagel A, Dougherty T, Fullaway CC, McIntosh BJ, Leow KX, Schwartz MS, Pavelchek C. Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning. Nature biotechnology. 2022 Apr;40(4):555-65."

    https://github.com/vanvalenlab/deepcell-tf/blob/master/deepcell/applications/application.py

    Args:
        image (numpy.array): Input image to resize.
        image_mpp (float): Microns per pixel for the ``image``.

    Returns:
        numpy.array: Input image resized if necessary to match ``model_mpp``
    """
    # Don't scale the image if mpp is the same or not defined
    # if image_mpp not in {None, self.model_mpp}:
    shape = image.shape
    scale_factor = image_mpp / 0.5
    new_shape = (int(shape[1] * scale_factor), int(shape[2] * scale_factor))
    image = resize(image, new_shape, data_format="channels_last")

    return image


def format_output_mesmer(output_list):
    """Takes list of model outputs and formats into a dictionary for better readability

    https://github.com/vanvalenlab/deepcell-tf/blob/master/deepcell/applications/mesmer.py

    Args:
        output_list (list): predictions from semantic heads

    Returns:
        dict: Dict of predictions for whole cell and nuclear.

    Raises:
        ValueError: if model output list is not len(4)
    """
    expected_length = 4
    if len(output_list) != expected_length:
        raise ValueError(
            "output_list was length {}, expecting length {}".format(
                len(output_list), expected_length
            )
        )

    formatted_dict = {
        "whole-cell": [output_list[0], output_list[1][..., 1:2]],
        "nuclear": [output_list[2], output_list[3][..., 1:2]],
    }

    return formatted_dict


def deep_watershed(
    outputs,
    radius=10,
    maxima_threshold=0.1,
    interior_threshold=0.01,
    maxima_smooth=0,
    interior_smooth=1,
    maxima_index=0,
    interior_index=-1,
    label_erosion=0,
    small_objects_threshold=0,
    fill_holes_threshold=0,
    pixel_expansion=None,
    maxima_algorithm="h_maxima",
    **kwargs,
):
    """Uses ``maximas`` and ``interiors`` to perform watershed segmentation.
    ``maximas`` are used as the watershed seeds for each object and
    ``interiors`` are used as the watershed mask.

    https://github.com/vanvalenlab/deepcell-toolbox/blob/master/deepcell_toolbox/deep_watershed.py

    Args:
        outputs (list): List of [maximas, interiors] model outputs.
            Use `maxima_index` and `interior_index` if list is longer than 2,
            or if the outputs are in a different order.
        radius (int): Radius of disk used to search for maxima
        maxima_threshold (float): Threshold for the maxima prediction.
        interior_threshold (float): Threshold for the interior prediction.
        maxima_smooth (int): smoothing factor to apply to ``maximas``.
            Use ``0`` for no smoothing.
        interior_smooth (int): smoothing factor to apply to ``interiors``.
            Use ``0`` for no smoothing.
        maxima_index (int): The index of the maxima prediction in ``outputs``.
        interior_index (int): The index of the interior prediction in
            ``outputs``.
        label_erosion (int): Number of pixels to erode segmentation labels.
        small_objects_threshold (int): Removes objects smaller than this size.
        fill_holes_threshold (int): Maximum size for holes within segmented
            objects to be filled.
        pixel_expansion (int): Number of pixels to expand ``interiors``.
        maxima_algorithm (str): Algorithm used to locate peaks in ``maximas``.
            One of ``h_maxima`` (default) or ``peak_local_max``.
            ``peak_local_max`` is much faster but seems to underperform when
            given regious of ambiguous maxima.

    Returns:
        numpy.array: Integer label mask for instance segmentation.

    Raises:
        ValueError: ``outputs`` is not properly formatted.
    """
    try:
        maximas = outputs[maxima_index]
        interiors = outputs[interior_index]
    except (TypeError, KeyError, IndexError):
        raise ValueError(
            "`outputs` should be a list of at least two " "NumPy arryas of equal shape."
        )

    valid_algos = {"h_maxima", "peak_local_max"}
    if maxima_algorithm not in valid_algos:
        raise ValueError(
            "Invalid value for maxima_algorithm: {}. "
            "Must be one of {}".format(maxima_algorithm, valid_algos)
        )

    total_pixels = maximas.shape[1] * maximas.shape[2]
    if maxima_algorithm == "h_maxima" and total_pixels > 5000**2:
        warnings.warn(
            "h_maxima peak finding algorithm was selected, "
            "but the provided image is larger than 5k x 5k pixels."
            "This will lead to slow prediction performance."
        )

    if maximas.shape[:-1] != interiors.shape[:-1]:
        raise ValueError(
            "All input arrays must have the same shape. "
            "Got {} and {}".format(maximas.shape, interiors.shape)
        )

    if maximas.ndim not in {4, 5}:
        raise ValueError(
            "maxima and interior tensors must be rank 4 or 5. "
            "Rank 4 is 2D data of shape (batch, x, y, c). "
            "Rank 5 is 3D data of shape (batch, frames, x, y, c)."
        )

    input_is_3d = maximas.ndim > 4

    # fill_holes is not supported in 3D
    if fill_holes_threshold and input_is_3d:
        warnings.warn("`fill_holes` is not supported for 3D data.")
        fill_holes_threshold = 0

    label_images = []
    for maxima, interior in zip(maximas, interiors):
        # squeeze out the channel dimension if passed
        maxima = nd.gaussian_filter(maxima[..., 0], maxima_smooth)
        interior = nd.gaussian_filter(interior[..., 0], interior_smooth)

        if pixel_expansion:
            fn = cube if input_is_3d else square
            interior = dilation(interior, footprint=fn(pixel_expansion * 2 + 1))

        # peak_local_max is much faster but has poorer performance
        # when dealing with more ambiguous local maxima
        if maxima_algorithm == "peak_local_max":
            coords = peak_local_max(
                maxima,
                min_distance=radius,
                threshold_abs=maxima_threshold,
                exclude_border=kwargs.get("exclude_border", False),
            )

            markers = np.zeros_like(maxima)
            slc = tuple(coords[:, i] for i in range(coords.shape[1]))
            markers[slc] = 1
        else:
            # Find peaks and merge equal regions
            fn = ball if input_is_3d else disk
            markers = h_maxima(image=maxima, h=maxima_threshold, footprint=fn(radius))

        markers = label(markers)
        label_image = watershed(
            -1 * interior, markers, mask=interior > interior_threshold, watershed_line=0
        )

        if label_erosion:
            label_image = erode_edges(label_image, label_erosion)

        # Remove small objects
        if small_objects_threshold:
            label_image = remove_small_objects(
                label_image, min_size=small_objects_threshold
            )

        # fill in holes that lie completely within a segmentation label
        if fill_holes_threshold > 0:
            label_image = fill_holes(label_image, size=fill_holes_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images.append(label_image)

    label_images = np.stack(label_images, axis=0)
    label_images = np.expand_dims(label_images, axis=-1)

    return label_images
