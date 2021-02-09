import matplotlib.pyplot as plt

from pathml.preprocessing.wsi import HESlide
from pathml.preprocessing.transforms_HandE import TissueDetectionHE
import cv2
from pathml.preprocessing.transforms import  BinaryThreshold, MedianBlur
from pathml.preprocessing.utils import RGB_to_HSV


tma_path = "./data/TMA/HE_Hamamatsu.ndpi"
tma = HESlide(tma_path)

# check level
# different level gives different level of details and number of contours
tma_data = tma.load_data(level = 5)
#plt.imshow(tma_data.image)
#plt.show()

tissue_detector = TissueDetectionHE()
detected = tissue_detector.apply(tma_data.image)
#plt.imshow(detected)
#plt.show()

#plot_mask(tma_data.image, detected)

one_channel = RGB_to_HSV(tma_data.image)
one_channel = one_channel[:, :, 1]
blur = MedianBlur(kernel_size=17)
blurred = blur.apply(one_channel)
#plt.imshow(blurred)

threshold = BinaryThreshold(use_otsu=False, threshold=30)
thresholded = threshold.apply(blurred)
#plt.imshow(thresholded)

############## try OpenCV

# use RETR_EXTERNAL for external contours
contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#contours = contours[:10]
len(contours)

#cv2.drawContours(tma_data.image, contours, -1, (0,255,0), 3)
#cv2.imshow('Contours', tma_data.image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# plot bounding boxes
ROI_number = 0
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w > 20 and h > 20:
        cv2.rectangle(tma_data.image, (x-5, y-5), (x + w +5, y + h+5), (36,255,12), 2)
        ROI = tma_data.image[y:y+h, x:x+w]
        ROI_number += 1

#cv2.imshow('image', tma_data.image)
cv2.imwrite('tma_seg.png', tma_data.image)
#cv2.waitKey()

# plot individual images
idx =0
for cnt in contours:
    idx += 1
    x,y,w,h = cv2.boundingRect(cnt)
    roi=tma_data.image[(y-3) : (y+h+3), (x-3) : (x+w+3)]
    cv2.imwrite('data/TMA/cores/' + str(idx) + '.jpg', roi)
    #img = cv2.rectangle(tma_data.image,(x,y),(x+w,y+h),(200,0,0),2)
#cv2.imshow('img',tma_data.image)

