import matplotlib.pyplot as plt

from pathml.preprocessing.wsi import HESlide
from pathml.preprocessing.transforms_HandE import TissueDetectionHE
from pathml.preprocessing.utils import plot_mask

tma_path = "./data/TMA/HE_Hamamatsu.ndpi"
tma = HESlide(tma_path)

# check level
# different level gives different level of details and number of contours
tma_data = tma.load_data(level = 5)
plt.imshow(tma_data.image)
plt.show()

tissue_detector = TissueDetectionHE()
detected = tissue_detector.apply(tma_data.image)
#plt.imshow(detected)
#plt.show()

#plot_mask(tma_data.image, detected)
#========================== testing TissueDectionHE one step at a time
import cv2
import numpy as np

from pathml.preprocessing.base_transforms import SegmentationTransform
from pathml.preprocessing.stains import StainNormalizationHE
from pathml.preprocessing.transforms import MorphOpen, MorphClose, BinaryThreshold, ForegroundDetection, MedianBlur, SuperpixelInterpolationSLIC
from pathml.preprocessing.utils import contour_centroid, sort_points_clockwise, RGB_to_HSV, RGB_to_GREY



one_channel = RGB_to_HSV(tma_data.image)
one_channel = one_channel[:, :, 1]
blur = MedianBlur(kernel_size=17)
blurred = blur.apply(one_channel)
#plt.imshow(blurred)

threshold = BinaryThreshold(use_otsu=False, threshold=30)
threshold = BinaryThreshold(use_otsu=True)
thresholded = threshold.apply(blurred)
plt.imshow(thresholded)

# CANCEL OUT THE NOISE WITHIN THE CORE
################### NOT CLEARLY SEPERATED
opening = MorphOpen(kernel_size=7, n_iterations=3)
opened = opening.apply(thresholded)
plt.imshow(opened)

closing = MorphClose(kernel_size=7, n_iterations=3)
closed = closing.apply(opened)
plt.imshow(closed)
################### NOT CLEARLY SEPERATED

############# changed min_region_size from 5000 to 100
foreground_detection = ForegroundDetection(min_region_size=100, max_hole_size=1500, outer_contours_only=False)
tissue_regions = foreground_detection.apply(thresholded)
plt.imshow(tissue_regions)

plot_mask(tma_data.image, tissue_regions) # tissue_regions is mask_in

#==========================testing plot_mask

mask_in = tissue_regions

############ change kernel size???
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(mask_in, kernel)
plt.imshow(dilated)

############# y, x is the boundary, need to change to boxes
# this is to find the edges by getting the intersection of masks and dilated
diff = np.logical_xor(dilated.astype(bool), mask_in.astype(bool)) # 0 is False, 1 is True
y, x = np.nonzero(diff)

fig, ax = plt.subplots()
ax.imshow(tma_data.image)
ax.scatter(x, y, color = 'red', marker = ".", s = 1)
ax.axis('off')

############## try OpenCV

# use RETR_EXTERNAL for external contours
contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
len(contours)

#cv2.drawContours(tma_data.image, contours, -1, (0,255,0), 3)
#cv2.imshow('Contours', tma_data.image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

idx =0
for cnt in contours:
    idx += 1
    x,y,w,h = cv2.boundingRect(cnt)
    roi=tma_data.image[y:y+h,x:x+w]
    #cv2.imwrite('data/TMA/TMAs/' + str(idx) + '.jpg', roi)
    cv2.rectangle(tma_data.image,(x,y),(x+w,y+h),(200,0,0),2)
cv2.imshow('img',tma_data.image)
