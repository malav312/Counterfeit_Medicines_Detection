from helper_methods import *

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os 
import matplotlib.pyplot as plt

# load the list of numpy arrays
vectors = []
dir_path = "../data/output/vectors/"
list_of_medicines = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
list_of_npy_files = []
for medicine in list_of_medicines: 
    med_dir_path = dir_path+medicine+"/"
    list_of_photos = os.listdir(med_dir_path)
    for photo in list_of_photos:
        list_of_npy_files.append(med_dir_path+photo)
# print(list_of_npy_files)
# list_of_npy_files = list_of_npy_files.sort()
for filename in list_of_npy_files:
    vector = np.load(filename)
    vectors.append(vector)
vector1list= [list_of_npy_files[0],list_of_npy_files[1],list_of_npy_files[2]]
vectors1, vectors2 = [],[]
for f in vector1list:
    vector = np.load(f)
    vectors1.append(vector)
vector2list = [list_of_npy_files[9],list_of_npy_files[8],list_of_npy_files[10]]
for f in vector2list:
    vector = np.load(f)
    vectors2.append(vector)

# print(vector1)
# print(vector2)

# convert the list of vectors to a numpy array
vectors = np.array(vectors)

# compute cosine similarities between all pairs of vectors
cos_sim = cosine_similarity(vectors1,vectors2)

# print the resulting matrix
# print(cos_sim)
fig, ax = plt.subplots()
im = ax.imshow(cos_sim, cmap='Greens', vmin=0.7, vmax=1)

ax.tick_params(axis='x', labelrotation=90)
cbar = ax.figure.colorbar(im, ax=ax)
plt.savefig("Acenac vs Ace.png")
