import cv2
import numpy as np

from pathml.preprocessing.base import SegmentationTransform
from pathml.preprocessing.stains import StainNormalizationHE
from pathml.preprocessing.transforms import MorphOpen, MorphClose, BinaryThreshold, ForegroundDetection, MedianBlur, \
    SuperpixelInterpolationSLIC
from pathml.preprocessing.utils import contour_centroid, sort_points_clockwise, RGB_to_HSV, RGB_to_GREY


class TissueDetectionHE(SegmentationTransform):
    """
    Detect tissue regions from H&E stained slide. Works by combining blurring, thresholding, and morphology operators.

    :param use_saturation: Whether to convert to HSV and use saturation channel for tissue detection.
        If False, convert from RGB to greyscale and use greyscale image_ref for tissue detection. Defaults to True.
    :type use_saturation: bool
    :param blur: Blur operation applied before binary thresholding. Helps reduce noise in input image_ref.
        If ``None``, uses ``MedianBlur(kernel_size = 17)``. Defaults to ``None``.
    :type blur: :class:`~pathml.preprocessing.base.ImageTransform`
    :param threshold: Binary thresholding operation. If ``None``, uses
        ``BinaryThreshold(use_otsu = False, threshold = 30)``. Defaults to ``None``.
    :type threshold: :class:`~pathml.preprocessing.base.SegmentationTransform`
    :param opening: Morphological opening transformation. Helps reduce noise from binary thresholding.
        If ``None``, uses ``MorphOpen(kernel_size = 7, n_iterations = 3)``. Defaults to ``None``.
    :type opening: :class:`~pathml.preprocessing.base.MaskTransform`
    :param closing: Morphological closing transformation. Helps reduce noise from binary thresholding.
        If ``None``, uses ``MorphClose(kernel_size = 7, n_iterations = 3)``. Defaults to ``None``.
    :type closing: :class:`~pathml.preprocessing.base.MaskTransform`
    :param foreground_detection: Foreground detection transform.
        If ``None``, uses ``ForegroundDetection()``. Defaults to ``None``.
    :type foreground_detection: :class:`~pathml.preprocessing.base.MaskTransform`
    """

    def __init__(
            self, use_saturation=True, blur=None, threshold=None, opening=None, closing=None, foreground_detection=None
    ):
        self.use_sat = use_saturation
        self.blur = MedianBlur(kernel_size = 17) if blur is None else blur
        self.threshold = BinaryThreshold(use_otsu = False, threshold = 30) if threshold is None else threshold
        self.opening = MorphOpen(kernel_size = 7, n_iterations = 3) if opening is None else opening
        self.closing = MorphClose(kernel_size = 7, n_iterations = 3) if closing is None else closing
        self.foreground_detection = ForegroundDetection() if foreground_detection is None else foreground_detection

    def apply(self, image):
        """
        Apply tissue detection to input image.

        :return: Mask indicating tissue regions.
        :rtype: np.ndarray
        """
        assert image.dtype == np.uint8, f"Input image dtype {image.dtype} must be np.uint8"
        assert image.shape[2] == 3, f"ERROR image input shape is {image.shape} "
        # first get single channel image_ref
        if self.use_sat:
            one_channel = RGB_to_HSV(image)
            one_channel = one_channel[:, :, 1]
        else:
            one_channel = RGB_to_GREY(image)

        blurred = self.blur.apply(one_channel)
        thresholded = self.threshold.apply(blurred)
        opened = self.opening.apply(thresholded)
        closed = self.closing.apply(opened)
        tissue_regions = self.foreground_detection.apply(closed)
        return tissue_regions


class BlackPenDetectionHE(SegmentationTransform):
    """
    Detect regions circled by pathologists in black pen on a H&E whole-slide image_ref.

    :param black_threshold: RGB threshold for labelling black pen marks. Any pixels below the thresholds will be
        considered pen marks. Defaults to (110, 110, 110).
    :type black_thresh: tuple of (R_thresh, G_thresh, B_thresh)
    :param opening: morphological opening transform to perform after thresholding.
        This helps remove noise from thresholding. If None, uses ``MorphOpen(kernel_size = 11, n_iterations = 3)``.
        Defaults to ``None``.
    :type opening: :class:`~pathml.preprocessing.base.MaskTransform`
    :param smoothing: Transform to smooth the output of segmentation.
        If ``None``, uses
        ``MorphOpen(n_iterations = 3, custom_kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (31, 31)))``.
        Defaults to ``None``.
    :return: binary mask indicating regions within pen marks.
    :rtype: np.ndarray
    """

    def __init__(
            self,
            black_threshold=(110, 110, 110),
            opening=None,
            smoothing=None,
    ):
        self.black_thresh = black_threshold
        self.opening = MorphOpen(kernel_size = 11, n_iterations = 3) if opening is None else opening
        self.smoothing = MorphOpen(
                n_iterations = 3,
                custom_kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (31, 31))
            ) if smoothing is None else smoothing

    def apply(self, image):
        assert image.dtype == np.uint8, f"Input image dtype {image.dtype} must be np.uint8"
        # apply thresholds
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        r_thresh, g_thresh, b_thresh = self.black_thresh
        black_pen = np.logical_and(r < r_thresh, g < g_thresh)
        black_pen = np.logical_and(black_pen, b < b_thresh)
        black_pen = black_pen.astype(np.uint8)
        opened = self.opening.apply(black_pen)
        # TODO: make the grid size adaptive based on size of inputs
        # apply a grid of 0s. Then we can treat as a dotted line and connect the dots.
        for i in np.arange(0, opened.shape[0], 100):
            opened[i:i + 50, :] = 0
        for j in np.arange(0, opened.shape[1], 100):
            opened[:, j:j + 75] = 0

        # detect contours of pen marks
        contours, hierarchy = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            # no valid contours found
            out = np.zeros_like(opened)
            return out

        contour_centroids = [contour_centroid(c) for c in contours if cv2.contourArea(c) > 10]

        if len(contours) == 0:
            # no valid contours found that meet area thresholds
            out = np.zeros_like(opened)
            return out

        contour_centroids = np.vstack([np.array(c) for c in contour_centroids])

        # sort to be in order around the circle
        contour_centroids = sort_points_clockwise(contour_centroids.astype(np.int))
        # return filled polygon of shape defined by centers of bounding boxes
        out = np.zeros_like(opened)
        out = cv2.fillPoly(out, pts = [contour_centroids], color = (255, 255, 255))

        out = self.smoothing.apply(out)
        return out


class BasicNucleusDetectionHE(SegmentationTransform):
    """
    Simple nucleus detection algorithm. Works by first separating hematoxylin channel, then doing interpolation using
    superpixels, and finally using binary thresholding.

    :param hematoxylin_separator: stain separator to extract hematoxylin channel
        If ``None``, uses ``StainNormalizationHE(target = "hematoxylin", stain_estimation_method = "vahadane")``.
        Defaults to ``None``.
    :type hematoxylin_separator: :class:`~pathml.preprocessing.base.ImageTransform`
    :param superpixel_interpolator: transform to perform superpixel interpolation after separating hematoxylin channel.
        If ``None``, uses ``SuperpixelInterpolationSLIC()``. Defaults to ``None``.
    :type superpixel_interpolator: :class:`~pathml.preprocessing.base.ImageTransform`
    :param threshold: Binary threshold transform. If ``None``, uses ``BinaryThreshold(use_otsu=True)``.
        Defaults to ``None``.
    :type threshold: :class:`~pathml.preprocessing.base.SegmentationTransform`


    References:
        Hu, B., Tang, Y., Eric, I., Chang, C., Fan, Y., Lai, M. and Xu, Y., 2018. Unsupervised learning for cell-level
        visual representation in histopathology images with generative adversarial networks. IEEE journal of
        biomedical and health informatics, 23(3), pp.1316-1328.
    """

    def __init__(
            self,
            hematoxylin_separator=None,
            superpixel_interpolator=None,
            threshold=None
    ):
        if hematoxylin_separator is None:
            self.hematoxylin_separator = StainNormalizationHE(
                target = "hematoxylin", stain_estimation_method = "vahadane")
        else:
            self.hematoxylin_separator = hematoxylin_separator
        if superpixel_interpolator is None:
            self.superpixel_interpolator = SuperpixelInterpolationSLIC()
        else:
            self.superpixel_interpolator = superpixel_interpolator

        if threshold is None:
            self.threshold = BinaryThreshold(use_otsu = True)
        else:
            self.threshold = threshold

    def apply(self, image):
        assert image.dtype == np.uint8, f"Input image dtype {image.dtype} must be np.uint8"
        im_hematoxylin = self.hematoxylin_separator.apply(image)
        im_interpolated = self.superpixel_interpolator.apply(im_hematoxylin)
        thresholded = self.threshold.apply(im_interpolated)
        # flip sign so that nuclei regions are TRUE (255)
        thresholded = ~thresholded
        return thresholded
