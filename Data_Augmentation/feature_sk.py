import os
import cv2
import pytesseract
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


os.environ['TF_FORCE_GPU_ALLOWED'] = '0'

# Function to get the text of a medicine strip


class getText:

    def get_text(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Perform OCR using pytesseract
        results = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
        print(text)

    def get_vector(image_path):
        # Load the ResNet50 model
        model = ResNet50(weights='imagenet', include_top=False,
                         input_shape=(224, 224, 3))

        # Read the image and resize it to (224, 224)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))

        # Preprocess the image for ResNet50
        img = preprocess_input(np.expand_dims(img, axis=0))

        # Get the features from the ResNet50 model
        features = model.predict(img)

        # Flatten the features to a vector of size 512
        vector = np.ravel(features)

        # Return the vector
        return vector

    # Function to get the coordinates of the text in an image
    def get_text_regions(image_path):
        # Read the image
        img = cv2.imread(image_path)

        # Remove background using image segmentation or background subtraction
        fgbg = cv2.createBackgroundSubtractorMOG2()
        fgmask = fgbg.apply(img)
        bg_removed = cv2.bitwise_and(img, img, mask=fgmask)

        # Apply filters to remove noise or improve contrast
        gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)

        # Detect regions likely to contain text using contour detection or edge detection
        canny = cv2.Canny(equalized, 50, 200)
        contours, hierarchy = cv2.findContours(
            canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500 and area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                text_regions.append((x, y, x+w, y+h))

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(thresh, kernel, iterations=1)
        erode = cv2.erode(dilate, kernel, iterations=1)

        regions = pytesseract.image_to_boxes(erode, lang='eng')

        coordinates = []
        for region in regions.splitlines():
            region = region.split(' ')
            coordinates.append((int(region[1]), int(
                region[2]), int(region[3]), int(region[4])))

        return coordinates

    def get_text_coordinates(image_path):
        # Read the image
        img = cv2.imread(image_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(thresh, kernel, iterations=1)

        boxes = pytesseract.image_to_boxes(dilate, lang='eng')

        coordinates = []
        for box in boxes.splitlines():
            box = box.split(' ')
            coordinates.append(
                (int(box[2]), int(box[1]), int(box[4]), int(box[3])))
        return coordinates

    def draw_rectangle(img, boxes):
        image = cv2.imread(img)
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]),
                          (box[2], box[3]), (0, 255, 0), 2)

        cv2.imshow("Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# cord = getText.get_text_regions(
#     "../input/meds/images/A2__Tablet/A2__Tablet2.jpg")
# print(cord)
# getText.draw_rectangle("../input/meds/images/A2__Tablet/A2__Tablet2.jpg", cord)
