import cv2
import pytesseract
import spacy
import sqlite3
import pandas as pd
# from fuzzywuzzy import fuzz
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from tqdm import tqdm
from functools import partial
import string

#Loading the English language model
nlp = spacy.load("en_core_web_sm")

# img = cv2.imread("Accept-SP_Tablet2 (1).jpg")

# #processing for the noise
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.medianBlur(gray, 3)
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Extract text from image using OCR
# text = pytesseract.image_to_string(gray)

# Use NER to identify medicine names from extracted text
# # doc = nlp(text)
# medicine_names = []
# for ent in doc.ents:
#     if ent.label_ == "MEDICINE":
#         medicine_names.append(ent.text)


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
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # print('Texts:')
 
    #area-----------
    max_area = 0
    max_area_name = ""
    text_list = []

    for text in texts:
        text_list.append('\n"{}"'.format(text.description))
        

        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]

        # vertices = (['({},{})'.format(vertex.x, vertex.y)
        #             for vertex in text.bounding_poly.vertices])
        
        #Case 1: Area using the vertices (Highest area might be the case) - WORKING
        area = 0
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            area += x1 * y2 - x2 * y1
        area /= 2
        if area > max_area:
            max_area = area
            # max_area_name = text.description

        #Case 2: 

    # text_list.append(max_area_name)
    # print("Name with highest area:", max_area_name)


    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return text_list
    
medicine_names = [] 

image_text = detect_text("Accept-SP_Tablet2 (1).jpg")
temp_df = pd.DataFrame(columns=['text'])
temp_df['text'] = image_text[0].replace("\n"," ").split()
temp_df = remove_punctuations(temp_df)
temp_df = lower_case(temp_df)
temp_df = remove_null(temp_df)


#verifying via sqlite
conn = sqlite3.connect("updated_processed_names.db")
c = conn.cursor()

validated_medicine_names = []
for name in temp_df:
    name_vec =nlp(name.lower()).vector
    # print("Dimensions", name_vec.shape)

    # Execute SQL query to check if name is present in database
    c.execute("SELECT TEXT FROM text WHERE TEXT=?", (name,))
    result = c.fetchone()
    if result is not None:
        validated_medicine_names.append(name)
        if len(validated_medicine_names) == 5: # Limit the number of validated medicine names to 5
            break
    else:
        print("Here")
        c.execute("SELECT TEXT FROM text")
        print("Also reached here")
        db_names = [row[0].strip() for row in c.fetchall() if row[0].strip()]
        print("Dimensions", np.array(db_names).shape)
        if not db_names:
            continue
        print("Before vec")
        # db_vecs = [nlp(db_name.lower()).vector for db_name in tqdm(db_names)]
        # db_vecs = list(python.apply(partial(nlp, disable=["parser", "tagger", "ner"]), [db_name.lower() for db_name in tqdm(db_names)]))

        db_vecs = np.load("db_vecs.npy")
        # np.save("db_vecs.npy",np.array(db_vecs))
        # print("Dimensions", np.array(db_vecs).shape)

        print("After vec")
        max_similiarity = -1
        print("Have i reached here")
        validated_name = None
        count = 0
        for db_name in db_names:
            similarity = np.max(cosine_similarity(db_vecs,list(np.array(name_vec).reshape(1, -1))))
            if similarity > max_similiarity:
                max_similiarity = similarity
                validated_name = db_name
                print("Reached here")
            if validated_name is not None:
                validated_medicine_names.append(validated_name)
                if len(validated_medicine_names) == 1: # Limit the number of validated medicine names to 5
                    break




# Close database connection
conn.close()

#Print validated medicine names
print(validated_medicine_names)
# detect_text("Accept-SP_Tablet2 (1).jpg")