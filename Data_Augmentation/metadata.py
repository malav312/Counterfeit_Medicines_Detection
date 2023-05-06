from features import getFeatures
import os
import cv2

import os
from PIL import Image

VALID_EXTENSIONS = ('.jpg')

def is_valid_image(filename):
    """Returns True if the file is a valid image file."""
    return os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS

def process_directory(dir_path):
    """Process directory and its contents."""
    image_path=[]
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path) and is_valid_image(filename):
            # process image file here
            image_path.append(file_path)
        
        elif os.path.isdir(file_path):
            image_path.extend(process_directory(file_path))
        return image_path

all_image_paths = []
# pass every directory path to process_directory function
for dirpath, dirnames, filenames in os.walk('../input/meds/images'):
    image_path = process_directory(dirpath)
    all_image_paths.append(image_path)

for image_path in all_image_paths:
    for image_p in image_path:
        img = cv2.imread(image_p)
        getFeatures.color_moments(img)