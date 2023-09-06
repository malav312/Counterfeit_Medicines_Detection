import spacy
import sqlite3
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from google.cloud import vision
import io
import string
from nltk.corpus import stopwords
from difflib import SequenceMatcher
from textblob import TextBlob
from fuzzywuzzy import process, fuzz
import re
from scipy.spatial import distance
from skimage.feature import graycomatrix, graycoprops
import pytesseract
import cv2
import json
import math
import csv

from identify_functions import wordList


def wordDict(image_path):
    df = wordList(image_path)
    word_list = df.values.flatten().tolist()
    returnDict = {'Word List': word_list}
    return returnDict


def color_moments(image_path):
    img = cv2.imread(image_path)
    channels = cv2.split(img)

    colour_features = []
    for channel in channels:
        moments = cv2.moments(channel)
        for i in range(3):
            for j in range(3):
                if i + j <= 2:
                    colour_features.append(moments['m{}{}'.format(i, j)] / moments['m00'])

    returnDict = {'Colour Features': colour_features}
    return returnDict


def texture_features(image_path):
        # Convert image to grayscale

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute GLCM features
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')

        returnDict = {'Contrast': contrast.flatten()[0],
                     'Dissimilarity': dissimilarity.flatten()[0], 
                     'Homogeneity': homogeneity.flatten()[0], 
                     'Energy':energy.flatten()[0], 
                     'Coorelation':correlation.flatten()[0]}
        return returnDict

def shape_features(image_path):
    # Convert image to grayscale
    img = cv2.imread(image_path)
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

    returnDict = {
    'Mean Area': mean_area,
    'Std Area': std_area,
    'Mean Perimeter': mean_perimeter,
    'Std Perimeter': std_perimeter,
    'Mean Aspect Ratio': mean_aspect_ratio,
    'Std Aspect Ratio': std_aspect_ratio,
    'Mean Centroid X': mean_centroid_x,
    'Std Centroid X': std_centroid_x,
    'Mean Centroid Y': mean_centroid_y,
    'Std Centroid Y': std_centroid_y
    }   
    # Return the computed features
    return returnDict

def pattern_features(image_path):
    img = cv2.imread(image_path)
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Compute the normalized histogram of the image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / np.sum(hist)

    # Compute the entropy of the histogram
    eps = np.finfo(float).eps
    entropy = -np.sum(hist_norm * np.log2(hist_norm + eps))
    entropylist = entropy.tolist()
    # entropylist = [float(x) if isinstance(x, np.float32) else x for x in entropylist]
    returnDict = {'Entropy': entropylist}

    # Return the computed feature
    return returnDict

def executeFeatures(image_path):
    wordList = wordDict(image_path)
    colour_moments = color_moments(image_path)
    texture = texture_features(image_path)
    shape = shape_features(image_path)
    pattern = pattern_features(image_path)

    returnDict = {'Text' : wordList, 'Color Moments': colour_moments, 'Texture':texture,'Shape':shape, 'Pattern':pattern}

    return returnDict



def is_valid_json(filename):
    VALID_EXTENSIONS = ('.json')
    """Returns True if the file is a valid image file."""
    return os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS

def process_directory(dir_path):
    """Process directory and its contents."""
    image_path=[]
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path) and is_valid_json(filename):
            # process image file here
            image_path.append(file_path)
        
        elif os.path.isdir(file_path):
            image_path.extend(process_directory(file_path))
    return image_path

def toJson(image_path):
    output_dir="fastapi/analysis_image"
    output_name = f"temp.json"
    output_path = os.path.join(output_dir, output_name)
    print("Outpath:",output_path)
    if os.path.exists(output_path):
        output_name = f"temp.json"
        output_path = os.path.join(output_dir, output_name)
        return 1
    with open(output_path, 'w') as f:
        json.dump(executeFeatures(image_path), f)
    return output_path

#Now checking Similarity
def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(list2)) 
    union = len(list1 + list2) - intersection 
    return intersection / union

def getMetric(json_file_1, json_file_2):
# json_file_1 = "metaanalysis/AB_Pas_N_Tablet/AB_Pas_N_Tablet_1.json"
# json_file_2 = "metaanalysis/AB_Pas_N_Tablet/AB_Pas_N_Tablet_2.json"
    with open(json_file_1, 'r') as f1:
        json_data1 = json.load(f1)

    text_dict1 = {
        "Word List": json_data1['Text']["Word List"]
    }
    colour_features_dict1 = {
        "Colour Features": json_data1["Color Moments"]["Colour Features"]
    }
    texture_dict1 =  json_data1["Texture"]
    shape_dict1 =  json_data1["Shape"]
    pattern_dict1 = json_data1["Pattern"]

    with open(json_file_2, 'r') as f2:
        json_data2 = json.load(f2)
    text_dict2 = {
        "Word List": json_data2['Text']["Word List"]
    }
    colour_features_dict2 = {
        "Colour Features": json_data2["Color Moments"]["Colour Features"]
    }
    texture_dict2 =  json_data2["Texture"]
    shape_dict2 =  json_data2["Shape"]
    pattern_dict2 = json_data2["Pattern"]

    #word list  using Jaccardian Distance
    word_list1 = text_dict1['Word List']
    word_list2 = text_dict2['Word List']
    dw = 0
    for l in word_list1:
        dw = 1-jaccard_similarity(word_list1, word_list2)

    #color list using Euclidean Distance
    color_list1 = colour_features_dict1['Colour Features']
    color_list2 = colour_features_dict2['Colour Features']
    dc=distance.euclidean(color_list1,color_list2)

    #texture list using Euclidean Distance
    texture_list1 = list(texture_dict1.values())
    texture_list2 = list(texture_dict2.values())
    dtx =distance.euclidean(texture_list1,texture_list2)

    #shape list using Euclidean Distance
    shape_list1 = list(shape_dict1.values())
    shape_list2 = list(shape_dict2.values())
    ds = distance.euclidean(shape_list1,shape_list2)

    #pattern list using Euclidean Distance
    pattern_list1 = list(pattern_dict1.values())
    pattern_list2 = list(pattern_dict2.values())
    dp = distance.euclidean(pattern_list1,pattern_list2)

    d = math.sqrt((dw**2)+(dc**2)+(dtx**2)+(ds**2)+(dp**2))

    return d

def find_value_by_name(name):
    with open('iqr_scores.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row if present

        for row in csv_reader:
            if row[0] == name:
                return row[1]

    return None

def comparision_metric(input_image_json_path, medicine_name):
    dir_path = "Data_Augmentation/metaanalysis"
    dir_path = os.path.join(dir_path,medicine_name)
    image_path_list = process_directory(dir_path)
    metric_list = []
    for json in image_path_list:
        d = getMetric(json, input_image_json_path)
        metric_list.append(d)
    average = sum(metric_list)/len(metric_list)
    getValue= find_value_by_name(medicine_name)
    print(getValue)
    value = float(getValue)
    if average < value:
        print("Yes")
        return "Authentic Medicine"
    else:
        print("No")
        return "Fake Medicine"

comparision_metric("Data_Augmentation/metaanalysis/AB_Pas_N_Tablet/AB_Pas_N_Tablet_2.json","A-Phyl_100_Capsule")