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
import warnings

warnings.filterwarnings("ignore")

def lower_case(df):
    df['text'] = df['text'].apply(str.lower)
    return df

def remove_punctuations(df):
    cleaned_text = []
    for index in tqdm(range(df.shape[0])):
        text = df['text'].iloc[index]

        word_tokens = text.split()
        
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in word_tokens]

        filtered_sentence = " ".join(stripped).strip()
        cleaned_text.append(filtered_sentence)
    df['text'] = np.array(cleaned_text)
    return df

def remove_null(df):
    if df['text'].isnull().sum() > 0:
        df.dropna(inplace = True)
    return df

def detect_text(path):
    """Detects text in the file."""
    
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
 
    text_list = []

    for text in texts:
        text_list.append('\n"{}"'.format(text.description))
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]

        # vertices = (['({},{})'.format(vertex.x, vertex.y)
        #             for vertex in text.bounding_poly.vertices])
        
        #Case 1: Area using the vertices (Highest area might be the case) - WORKING
        # area = 0
        # for i in range(len(vertices)):
        #     x1, y1 = vertices[i]
        #     x2, y2 = vertices[(i + 1) % len(vertices)]
        #     area += x1 * y2 - x2 * y1
        # area /= 2
        # if area > max_area:
        #     max_area = area

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    return text_list

def detect_text_from_image(path):
    image_text = detect_text(path)
    temp_df = pd.DataFrame(columns=['text'])
    temp_df['text'] = image_text[0].replace("\n"," ").split()
    temp_df = remove_punctuations(temp_df)
    temp_df = lower_case(temp_df)
    temp_df = remove_null(temp_df)

    return temp_df

def stop_words(df):
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    with open('Data_Augmentation/selected_words.txt', 'r') as f:
        stop_words = [word.strip().replace('"', '') for line in f.readlines() for word in line.split(',')]

    stop_words.extend(['composition','tablet','capsule','capsules','tablets','warning','dosage','direction','directions','use','uses', 'physician','coated','film'])
    stop.extend(stop_words)
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return df

def cleanDataframeEnglish(df,column_name):
  cleanedDf = pd.DataFrame(columns=['words'])
  for index, row in df.iterrows():
      word = row[column_name]
      if word.isalnum() and not word.isspace() and word.isascii():
        new_row = pd.DataFrame.from_records([{'words':word}])
        # print("New Row",new_row)
        cleanedDf = pd.concat([cleanedDf,new_row],ignore_index=True)
        # print("Iteration",cleanedDf)

  return cleanedDf

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_similarity_score(df):
    conn = sqlite3.connect('Data_Augmentation/removedwords.db')
    c = conn.cursor()
    c.execute("SELECT TEXT FROM text;")
    db_names = [row[0].rstrip() for row in c.fetchall() if row[0].strip()]
    db_names
    db_df = pd.DataFrame(db_names, columns = ['text'])
    db_df
    for i, row in df.iterrows():
    # Find the most similar text in the db_name column of db_df
        max_similarity = 0
        for j, row2 in db_df.iterrows():
            # Check if the value is a float
            if isinstance(row['words'], float):
                continue
            similarity = similar(row['words'], row2['text'])
            if similarity > max_similarity:
                max_similarity = similarity
                max_row = row2
        # Print the results
        if not isinstance(row['words'], float):
            print(f"Text '{row['words']}' has the most similar text '{max_row['text']}' with a similarity score of {max_similarity}")


def wordList(image_path):
    
    df = detect_text_from_image(image_path)
    df = stop_words(df)
    df = cleanDataframeEnglish(df,'text')
    df = df.drop_duplicates(subset=['words'])
    df = df.reset_index(drop=True)
    return df

def identify(image_path):
    df = wordList(image_path)
    get_similarity_score(df)
    

    # return df

identify('Data_Augmentation/Accept-SP_Tablet2 (1).jpg')
