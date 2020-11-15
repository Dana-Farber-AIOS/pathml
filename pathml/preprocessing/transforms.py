import cv2
import numpy as np

from pathml.preprocessing.base import ImageTransform, SegmentationTransform, MaskTransform
from pathml.preprocessing.utils import RGB_to_GREY


class MedianBlur(ImageTransform):
    """
    Median blur transform

    :param kernel_size: Width of kernel. Must be an odd number. Defaults to 5.
    """
    def __init__(self, kernel_size=5):
        assert kernel_size % 2 == 1, "kernel_size must be an odd number"
        self.kernel_size = kernel_size

    def apply(self, image):
        """Apply transform to input image"""
        assert image.dtype == np.uint8, f"Input image dtype {image.dtype} must be np.uint8"
        out = cv2.medianBlur(image, ksize = self.kernel_size)
        return out


class GaussianBlur(ImageTransform):
    """
    Gaussian blur transform

    :param kernel_size: Width of kernel. Must be an odd number. Defaults to 5.
    :param sigma: Variance of Gaussian kernel. Variance is assumed to be equal in X and Y axes. Defaults to 5.
    """
    def __init__(self, kernel_size=5, sigma=5):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def apply(self, image):
        """Apply transform to input image"""
        assert image.dtype == np.uint8, f"Input image dtype {image.dtype} must be np.uint8"
        out = cv2.GaussianBlur(
            image,
            ksize = (self.kernel_size, self.kernel_size),
            sigmaX = self.sigma,
            sigmaY = self.sigma
        )
        return out


class BoxBlur(ImageTransform):
    """
    Box blur transform. Averages each pixel with nearby pixels.

    :param kernel_size: Width of kernel. Defaults to 5.
    """
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def apply(self, image):
        """Apply transform to input image"""
        assert image.dtype == np.uint8, f"Input image dtype {image.dtype} must be np.uint8"
        out = cv2.boxFilter(image, ksize = (self.kernel_size, self.kernel_size), ddepth = -1)
        return out


class BinaryThreshold(SegmentationTransform):
    """
    Binary thresholding transform. If input image has 3 channels, it is assumed to be RGB and is
    first converted to greyscale.

    :param use_otsu: Whether to use Otsu's method to automatically determine optimal threshold. Defaults to True.
    :type use_otsu: bool, optional
    :param threshold: Specified threshold. Ignored if use_otsu==True. Defaults to 0.
    :type threshold: int, optional

    References:
        Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE transactions on systems,
        man, and cybernetics, 9(1), pp.62-66.
    """
    def __init__(self, use_otsu=True, threshold=0):
        self.threshold = threshold
        self.max_value = 255
        self.use_otsu = use_otsu
        if use_otsu:
            self.type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        else:
            self.type = cv2.THRESH_BINARY

    def apply(self, image):
        """
        Apply binary thresholding to input image_ref. If input image_ref has 3 channels, it is assumed to be RGB and is
        converted to greyscale.

        :param image: Input image
        :return: Binary mask result of thresholding
        """
        assert image.dtype == np.uint8, f"Input image dtype {image.dtype} must be np.uint8"
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = RGB_to_GREY(image)
            elif image.shape[2] == 1:
                image = image.squeeze(axis = 2)
            else:
                raise Exception(f"Input shape has {image.ndim} dimensions but third dimension {image.shape[2]} "
                                f"must be either 1 or 3")
        elif image.ndim != 2:
            raise Exception(f"Error: input image has {image.ndim} dimensions. Must have either 2 or 3 dimensions.")

        _, thresholded_mask = cv2.threshold(
            src = image,
            thresh = self.threshold,
            maxval = self.max_value,
            type = self.type
        )
        return thresholded_mask.astype(np.uint8)


class MorphOpen(MaskTransform):
    """
    Morphological opening transform.

    :param kernel_size: Size of kernel for default square kernel. Ignored if a custom kernel is specified.
        Defaults to 5.
    :type kernel_size: int, optional
    :param n_iterations: Number of opening operations to perform. Defaults to 1.
    :type n_iterations: int, optional
    :param custom_kernel: Optionally specify a custom kernel to use instead of default square kernel.
    :type custom_kernel: np.ndarray
    """
    def __init__(self, kernel_size=5, n_iterations=1, custom_kernel=None):

        self.kernel_size = kernel_size
        self.n_iterations = n_iterations
        self.custom_kernel = custom_kernel

    def apply(self, mask):
        """Apply transform to input mask"""
        assert mask.dtype == np.uint8, f"Input image dtype {mask.dtype} must be np.uint8"
        if self.custom_kernel is None:
            k = np.ones(self.kernel_size, dtype = np.uint8)
        else:
            k = self.custom_kernel
        out = cv2.morphologyEx(
            src = mask,
            kernel = k,
            op = cv2.MORPH_OPEN,
            iterations = self.n_iterations
        )
        return out


class MorphClose(MaskTransform):
    """
    Morphological closing transform.

    :param kernel_size: Size of kernel for default square kernel. Ignored if a custom kernel is specified.
        Defaults to 5.
    :param n_iterations: Number of opening operations to perform. Defaults to 1.
    :param custom_kernel: Optionally specify a custom kernel to use instead of default square kernel.
    """
    def __init__(self, kernel_size=5, n_iterations=1, custom_kernel=None):
        self.kernel_size = kernel_size
        self.n_iterations = n_iterations
        self.custom_kernel = custom_kernel

    def apply(self, mask):
        """Apply transform to input mask"""
        assert mask.dtype == np.uint8, f"Input image dtype {mask.dtype} must be np.uint8"
        if self.custom_kernel is None:
            k = np.ones(self.kernel_size, dtype = np.uint8)
        else:
            k = self.custom_kernel
        out = cv2.morphologyEx(
            src = mask,
            kernel = k,
            op = cv2.MORPH_CLOSE,
            iterations = self.n_iterations
        )
        return out


class ForegroundDetection(MaskTransform):
    """
    Foreground detection for binary masks. Identifies regions that have a total area greater than
    specified threshold. Supports including holes within foreground regions, or excluding holes
    above a specified area threshold.

    :param min_region_size: Minimum area of detected foreground regions, in pixels. Defaults to 5000.
    :type min_region_size: int, optional
    :param max_hole_size: Maximum size of allowed holes in foreground regions, in pixels.
        Ignored if outer_contours_only=True. Defaults to 1500.
    :type max_hole_size: int, optional
    :param outer_contours_only: If true, ignore holes in detected foreground regions. Defaults to False.
    :type outer_contours_only: bool, optional

    References:
        Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M. and Mahmood, F., 2020. Data Efficient and
        Weakly Supervised Computational Pathology on Whole Slide Images. arXiv preprint arXiv:2004.09666.
    """

    def __init__(self, min_region_size=5000, max_hole_size=1500, outer_contours_only=False):
        self.min_region_size = min_region_size
        self.max_hole_size = max_hole_size
        self.outer_contours_only = outer_contours_only

    def apply(self, mask):
        """Apply transformation to input mask."""
        assert mask.dtype == np.uint8, f"Input image dtype {mask.dtype} must be np.uint8"
        mode = cv2.RETR_EXTERNAL if self.outer_contours_only else cv2.RETR_CCOMP
        contours, hierarchy = cv2.findContours(mask.copy(), mode = mode, method = cv2.CHAIN_APPROX_NONE)

        if hierarchy is None:
            # no contours found --> return empty mask
            mask_out = np.zeros_like(mask)
        elif self.outer_contours_only:
            # remove regions below area threshold
            contour_thresh = np.array([cv2.contourArea(c) > self.min_region_size for c in contours])
            final_contours = np.array(contours)[contour_thresh]
            out = np.zeros_like(mask, dtype = np.int8)
            # fill contours
            cv2.fillPoly(out, final_contours, 255)
            mask_out = out
        else:
            # separate outside and inside contours (region boundaries vs. holes in regions)
            # find the outside contours by looking for those with no parents (4th column is -1 if no parent)
            hierarchy = np.squeeze(hierarchy, axis=0)
            outside_contours = hierarchy[:, 3] == -1
            hole_contours = ~outside_contours

            # outside contours must be above min_tissue_region_size threshold
            contour_size_thresh = [cv2.contourArea(c) > self.min_region_size for c in contours]
            # hole contours must be above area threshold
            hole_size_thresh = [cv2.contourArea(c) > self.max_hole_size for c in contours]
            # holes must have parents above area threshold
            hole_parent_thresh = [p in np.argwhere(contour_size_thresh).flatten() for p in hierarchy[:, 3]]

            # convert to np arrays so that we can do vectorized '&'. see: https://stackoverflow.com/a/22647006
            contours = np.array(contours)
            outside_contours = np.array(outside_contours)
            hole_contours = np.array(hole_contours)
            contour_size_thresh = np.array(contour_size_thresh)
            hole_size_thresh = np.array(hole_size_thresh)
            hole_parent_thresh = np.array(hole_parent_thresh)

            final_outside_contours = contours[outside_contours & contour_size_thresh]
            final_hole_contours = contours[hole_contours & hole_size_thresh & hole_parent_thresh]

            # now combine outside and inside contours into final mask
            out1 = np.zeros_like(mask, dtype = np.int8)
            out2 = np.zeros_like(mask, dtype = np.int8)
            # fill outside contours, inside contours, then subtract
            cv2.fillPoly(out1, final_outside_contours, 255)
            cv2.fillPoly(out2, final_hole_contours, 255)
            mask_out = (out1 - out2)

        return mask_out.astype(np.uint8)


class SuperpixelInterpolationSLIC(ImageTransform):
    """
    Divide input image into superpixels using SLIC algorithm, then interpolate each superpixel.
    SLIC superpixel algorithm described in Achanta et al. 2012.

    :param blur: Blur transform to apply. If ``None``, uses ``GaussianBlur()``. Defaults to ``None``.
    :type blur: :class:`~pathml.preprocessing.base.ImageTransform`
    :param superpixel_size: region_size parameter used for superpixel creation
    :type superpixel_size: int
    :param num_iter: Number of iterations to run SLIC algorithm
    :type num_iter: int
    :param threshold: Binary threshold transform. If ``None``, uses ``BinaryThreshold(use_otsu = True)``.
        Defaults to ``None``.
    :type threshold: :class:`~pathml.preprocessing.base.SegmentationTransform`

    References:
        Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P. and Süsstrunk, S., 2012. SLIC superpixels compared to
        state-of-the-art superpixel methods. IEEE transactions on pattern analysis and machine intelligence, 34(11),
        pp.2274-2282.
    """
    def __init__(self,
                 blur=None,
                 superpixel_size=10,
                 num_iter=30,
                 threshold=None,
                 ):
        self.blur = GaussianBlur() if blur is None else blur
        self.superpixel_size = superpixel_size
        self.num_iter = num_iter
        self.threshold = BinaryThreshold(use_otsu = True) if threshold is None else threshold

    @staticmethod
    def interpolate_superpixel_slic(image, slic):
        """
        After creating superpixels with SLIC method, interpolate each superpixel with average color.

        :param image: Original RGB image
        :type image: np.ndarray
        :param slic: ``slic`` class from openCV.
            Output of ``cv2.ximgproc.createSuperpixelSLIC().iterate()``
        """
        labels = slic.getLabels()
        nlabels = slic.getNumberOfSuperpixels()

        out = image.copy()

        # TODO this could be a lot more efficient probably by applying over channels instead of looping
        for i in range(nlabels):
            mask = labels == i
            for c in range(3):
                av = np.mean(image[:, :, c][mask])
                out[:, :, c][mask] = av
        return out

    def apply(self, image):
        """Apply transform to input image"""
        assert image.dtype == np.uint8, f"Input image dtype {image.dtype} must be np.uint8"
        blurred = self.blur.apply(image)

        # initialize slic class and iterate
        slic = cv2.ximgproc.createSuperpixelSLIC(image = blurred, region_size = self.superpixel_size)
        slic.iterate(num_iterations = self.num_iter)
        interp = self.interpolate_superpixel_slic(image = image, slic = slic)
        return interp
