import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os
import pytesseract
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from skimage.feature import graycomatrix, graycoprops


# Path to the input image
image_path = '../input/meds/images/AB-Flo_Capsule/AB-Flo_Capsule1.jpg'
image_path2 = '../input/meds/images/AB-Flo_Capsule/AB-Flo_Capsule1.jpg'
# image_path2 = '../input/meds/images/A2__Tablet/A2__Tablet0.jpg'
# Read the input image using OpenCV
img = cv2.imread(image_path)
img2 =cv2.imread(image_path2)

class getFeatures:

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
    
    def get_text(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Perform OCR using pytesseract
        results = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
        print(text)

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


    def color_moments(img):
        channels = cv2.split(img)

        colour_features = []
        for channel in channels:
            moments = cv2.moments(channel)
            for i in range(3):
                for j in range(3):
                    if i + j <= 2:
                        colour_features.append(moments['m{}{}'.format(i, j)] / moments['m00'])
        print(colour_features)

        return colour_features


    def texture_features(img):
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute GLCM features
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')

        # Return the computed features
        return list(contrast.flatten()) + list(dissimilarity.flatten()) + list(homogeneity.flatten()) + list(energy.flatten()) + list(correlation.flatten())

    def shape_features(img):
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Compute the contours of the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute the area, perimeter, and aspect ratio of each contour
        areas = []
        perimeters = []
        aspect_ratios = []
        centroid_xs = []
        centroid_ys = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            centroid_x = x + w/2
            centroid_y = y + h/2
            areas.append(area)
            perimeters.append(perimeter)
            aspect_ratios.append(aspect_ratio)
            centroid_xs.append(centroid_x)
            centroid_ys.append(centroid_y)

        # Compute the mean and standard deviation of the computed features
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        mean_perimeter = np.mean(perimeters)
        std_perimeter = np.std(perimeters)
        mean_aspect_ratio = np.mean(aspect_ratios)
        std_aspect_ratio = np.std(aspect_ratios)
        mean_centroid_x = np.mean(centroid_xs)
        std_centroid_x = np.std(centroid_xs)
        mean_centroid_y = np.mean(centroid_ys)
        std_centroid_y = np.std(centroid_ys)

        # Return the computed features
        return [mean_area, std_area, mean_perimeter, std_perimeter, mean_aspect_ratio, std_aspect_ratio, mean_centroid_x, std_centroid_x, mean_centroid_y, std_centroid_y]


    def pattern_features(img):
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Compute the normalized histogram of the image
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / np.sum(hist)

        # Compute the entropy of the histogram
        eps = np.finfo(float).eps
        entropy = -np.sum(hist_norm * np.log2(hist_norm + eps))

        # Return the computed feature
        return [entropy]

# def all_features(img):
#     color_moments_features = color_moments(img)
#     texture_features_list = texture_features(img)
#     shape_features_list = shape_features(img)
#     pattern_features_list = pattern_features(img)


#     return f"Colour Moments {color_moments_features}\n Texture Feature{texture_features_list}\n Shape Feature{shape_features_list}\n Pattern Feature{pattern_features_list}"


# Test the function
# all_features_list = all_features(img)
# image1_feature= color_moments(img)
# image2_feature= color_moments(img2)
# image1_feature= shape_features(img)
# image2_feature= shape_features(img2)




# similarity = np.dot(image1_feature, image2_feature) / (np.linalg.norm(image1_feature) * np.linalg.norm(image2_feature))
# print(similarity)
# metadata = {'img': {'similarity': similarity}, 'img2': {'similarity': similarity}}

# print(all_features_list)