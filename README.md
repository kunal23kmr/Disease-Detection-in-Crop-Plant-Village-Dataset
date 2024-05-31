# Plant Leaf Disease Detection Using CNN and SVM

### Overview
This project focuses on the detection of plant leaf diseases using two different machine learning models: Convolutional Neural Networks (CNN) and Support Vector Machines (SVM). The dataset used for this project contains images of plant leaves with various diseases. The goal is to preprocess these images, train the models, and compare their performance.

### Key Components
1. Preprocessing
Preprocessing is a crucial step in preparing the images for model training. The remove_bg function is used to enhance the contrast and remove the background of the leaf images, which helps in highlighting the diseased areas.

2. Models
CNN (Convolutional Neural Network): A deep learning model that is highly effective for image classification tasks.
SVM (Support Vector Machine): A classical machine learning model used for classification tasks, which works well with a well-defined feature set.
Dataset
The dataset consists of 87,000 images of plant leaves. The images are preprocessed using the remove_bg function before being used to train the models.

### Code
Preprocessing Function
The remove_bg function performs the following steps:

    1. Converts the image to grayscale.
    2. Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast.
    3. Uses Gaussian Blur to smooth the image.
    4. Applies Otsu's thresholding for binary segmentation.
    5. Performs morphological closing to fill gaps.
    6. Extracts the leaf contour and creates a mask to remove the background.



### Python code:
import cv2
import numpy as np
from skimage import exposure

def remove_bg(main_img):
    # Preprocessing
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (250, 250))

    gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = exposure.equalize_adapthist(gs, kernel_size=None, clip_limit=0.015, nbins=256)

    # Convert the float64 image to 8-bit
    clahe_uint8 = (clahe * 255).astype(np.uint8)

    # Adjusting contrast in the enhanced image
    contrasted_image = np.clip(clahe_uint8, 0, 255)

    blur = cv2.GaussianBlur(contrasted_image, (35, 35), 0)

    return_otsu_threshould, im_bw_otsu = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((35, 35), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
    # Contour
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Finding the correct leaf contour from the list of contours

    def find_contour(cnts):
        contains = []
        y_ri, x_ri, _ = img.shape
        for cc in cnts:
            yn = cv2.pointPolygonTest(cc, (x_ri//2, y_ri//2), False)
            contains.append(yn)
            val = [contains.index(temp) for temp in contains if temp > 0]
            if len(val) < 1:
                return -1
        return val[0]

    index = find_contour(contours)

    if index == -1:
        return img

    cnt = contours[index]
    # Create a black image to draw contours
    black_img = np.zeros_like(img)
    # Draw contour on the black image
    mask = cv2.drawContours(black_img, [cnt], 0, (255, 255, 255), -1)
    img = cv2.bitwise_and(img, mask)

    return img
