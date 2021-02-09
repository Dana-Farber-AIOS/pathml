import openslide
import cv2
import os

from pathml.preprocessing.wsi import HESlide
from pathml.preprocessing.transforms import  BinaryThreshold, MedianBlur
from pathml.preprocessing.utils import RGB_to_HSV


class TMA(HESlide):
    """
    Class for Tissue MicroArray slides, based on 'HESlide'`
    """
    def __init__(self, path):
        super().__init__(path)
        self.slide = openslide.open_slide(path)

    def apply(self):
        raise NotImplementedError


class TMASegmentation(HESlide):
    """
    Object for Tissue Microarray multi-cores segmentation.
    Outputs can be the original HE slide with TMA core bounding boxes, or individual TMA cores.

    Reference:
    contour_retrieval_mode: https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
    contour_approximation_method: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
    """

    def __init__(self, kernel_size = None,
                       threshold = None,
                       use_otsu = None,
                       contour_retrieval_mode = None,
                       contour_approximation_method = None):

        self.kernel_size = 17 if kernel_size is None else kernel_size
        self.threshold = 30 if threshold is None else threshold
        self.use_otsu = False if use_otsu is None else use_otsu
        self.contour_retrieval_mode = cv2.RETR_EXTERNAL if contour_retrieval_mode is None else contour_retrieval_mode
        self.contour_approximation_method = cv2.CHAIN_APPROX_NONE if contour_approximation_method is None else contour_approximation_method

    def apply(self, image):
        """
        Apply contour detection to the input TMA image.

        :return: a list of all contours. Each individual contour is a numpy array of (x,y) coordinates of boundary points of the object
        :rtype: list
        """
        one_channel = RGB_to_HSV(image)
        one_channel = one_channel[:, :, 1]
        blur = MedianBlur(self.kernel_size)
        blurred = blur.apply(one_channel)

        threshold = BinaryThreshold(self.use_otsu, self.threshold)
        thresholded = threshold.apply(blurred)

        # finding contours
        contours, hierarchy = cv2.findContours(thresholded, self.contour_retrieval_mode, self.contour_approximation_method)
        #cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        return contours


class BoundingBox():
    """
    Object for plotting bounding boxes around TMA cores. Save the plot to a specified path.

    :param minimum_width: The minimal width of a TMA core to be considered as a valid core.
    :type minimum_width: int
    :param minimum_height: The minimal height of a TMA core to be considered as a valid core.
    :type minimum_height: int
    :param width_adjustment: adjust the size of the bounding box by width.
    :type width_adjustment: int
    :param height_adjustment: adjust the size of the bounding box by height.
    :type height_adjustment: int
    """

    def __init__(self, minimum_width = None,
                       minimum_height = None,
                       width_adjustment = None,
                       height_adjustment = None):

        self.minimum_width = 20 if minimum_width is None else minimum_width
        self.minimum_height = 20 if minimum_height is None else minimum_height
        self.width_adjustment = 5 if width_adjustment is None else width_adjustment
        self.height_adjustment = 5 if height_adjustment is None else height_adjustment


    def apply(self, image, contours, save_path):
        """
        Plot bounding boxes to the individual cores in TMA slides. Save the bounding box plot to a specified path.

        :param image: The TMA slide image.
        :type image: numpy.ndarray
        :param contours: a list of all contours.
            Each individual contour is a numpy array of (x,y) coordinates of boundary points of the object.
        :type contours: list
        :param save_path: Path for saving the bounding box plot.
        :type save_path: str

        :return: The path where the bounding box plot is saved.
        :rtype: str
        """

        ROI_number = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > self.minimum_width and h > self.minimum_height:
                cv2.rectangle(image, (x - self.width_adjustment, y - self.height_adjustment),
                                    (x + w + self.width_adjustment, y + h + self.height_adjustment),
                                    (36, 255, 12), 2)
                ROI_number += 1

        cv2.imwrite(save_path, image)
        print(save_path)

class CoreSeparation():
    """
    Object for plotting separating TMA cores and saving all cores to a specified directory.

    :param minimum_width: The minimal width of a TMA core to be considered as a valid core.
    :type minimum_width: int
    :param minimum_height: The minimal height of a TMA core to be considered as a valid core.
    :type minimum_height: int
    :param width_adjustment: adjust the size of the individual cores by width.
    :type width_adjustment: int
    :param height_adjustment: adjust the size of the individual cores by height.
    :type height_adjustment: int
    """


    def __init__(self, minimum_width = None,
                       minimum_height = None,
                       width_adjustment = None,
                       height_adjustment = None):

        self.minimum_width = 20 if minimum_width is None else minimum_width
        self.minimum_height = 20 if minimum_height is None else minimum_height
        self.width_adjustment = 3 if width_adjustment is None else width_adjustment
        self.height_adjustment = 3 if height_adjustment is None else height_adjustment


    def apply(self, image, contours, save_directory):
        """
        Plot individual TMA cores and save them to a specified directory.

        :param image: The TMA slide image.
        :type image: numpy.ndarray
        :param contours: a list of all contours.
            Each individual contour is a numpy array of (x,y) coordinates of boundary points of the object.
        :type contours: list
        :param save_directory: Directory path for saving the individual TMA cores.
        :type save_directory: str

        :return: The path where the bounding box plot is saved.
        :rtype: str
        """
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

            idx = 0
            for cnt in contours:
                idx += 1
                x, y, w, h = cv2.boundingRect(cnt)
                if w > self.minimum_width and h > self.minimum_height:
                    roi = image[(y - self.width_adjustment): (y + h + self.height_adjustment),
                                (x - self.width_adjustment): (x + w + self.height_adjustment)]
                    cv2.imwrite(save_directory + str(idx) + '.jpg', roi)
        print(save_directory)
