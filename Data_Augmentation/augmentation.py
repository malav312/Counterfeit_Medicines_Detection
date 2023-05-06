import os
import tqdm
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter


class image_preprocessing:
    def __init__(self, df, dir_name):
        self.df = df
        self.dir_name = dir_name
    
    def get_df(self):
        return self.df
    
    def resize(self, img_path):
        # for img in tqdm(self.df['img']):
        print("here")
        _image = Image.open(os.path.join(img_path)).convert('RGB')
        _image = _image.resize((224, 224))
        _image.save(os.path.join(img_path))
        

    def __rotate(self, img_path):
        print("Rotating",img_path)
        new_file_name = img_path.strip('.jpg') + "_rotated.jpg"
        path = os.path.join(img_path)
        
        _image = np.array(Image.open(path).convert('RGB'))
        angle = tf.random.uniform([], minval=-60, maxval=60)
        _image = tf.keras.preprocessing.image.random_rotation(_image, angle, row_axis=0, col_axis=1, channel_axis=2)
        _image = Image.fromarray(_image)
        _image.save(os.path.join(new_file_name))
        
        # temp_df = self.df.loc[self.df['img_path'] == img_path]
        # text = temp_df.loc[temp_df.iloc[0].name, 'text']
        # id = temp_df.loc[temp_df.iloc[0].name, 'id']
        # if 'label' in self.df.columns:
        #     label = temp_df.loc[temp_df.iloc[0].name, 'label']
        #     self.df.loc[len(self.df.index)] = [id, new_file_name, label, text]
        # else:
        #     self.df.loc[len(self.df.index)] = [id, new_file_name, text]

    def __gaussian_noise(self, img):
        # new_file_name = "img/" + img.strip(".pngim") + "_augmented.png"
        print("Gaussian",img)

        new_file_name = img.strip('.jpg') + "_gaussian_noise.jpg"
        
        path = os.path.join(img)
        
        _image = np.array(Image.open(path).convert('RGB'))
        noise = tf.random.normal(shape=tf.shape(_image), mean=0, stddev=35, dtype=tf.float64)
        _image = np.clip(_image + noise, 0, 255)
        # _image = _image*255
        _image = np.array(_image, dtype=np.uint8)
        _image = Image.fromarray(_image)
        _image.save(os.path.join(new_file_name))
        
        # temp_df = self.df.loc[self.df['img'] == img]
        # text = temp_df.loc[temp_df.iloc[0].name, 'text']
        # id = temp_df.loc[temp_df.iloc[0].name, 'id']
        # if 'label' in self.df.columns:
        #     label = temp_df.loc[temp_df.iloc[0].name, 'label']
        #     self.df.loc[len(self.df.index)] = [id, new_file_name, label, text]
        # else:
        #     self.df.loc[len(self.df.index)] = [id, new_file_name, text]

    def __blur(self, img):
        print("Blur",img)
        new_file_name = img.strip('.jpg') + "_blur.jpg"
        path = os.path.join(img)
        
        _image = (Image.open(path).convert('RGB'))
        _image = _image.filter(ImageFilter.GaussianBlur(radius=0.8))
        _image.save(os.path.join(new_file_name))
        
        # temp_df = self.df.loc[self.df['img'] == img]
        # text = temp_df.loc[temp_df.iloc[0].name, 'text']
        # id = temp_df.loc[temp_df.iloc[0].name, 'id']
        # if 'label' in self.df.columns:
        #     label = temp_df.loc[temp_df.iloc[0].name, 'label']
        #     self.df.loc[len(self.df.index)] = [id, new_file_name, label, text]
        # else:
        #     self.df.loc[len(self.df.index)] = [id, new_file_name, text]

    def augmentation(self,img_path):
            # Select a random augmentation
            self.__rotate(img_path)
            self.__gaussian_noise(img_path)
            self.__blur(img_path)

#read csv
import pandas as pd
df = pd.read_csv('temp.csv')

for index, row in df.iterrows():
    # Access the first column of the current row using its label or index
    first_column_value = row[0]  # or row['column_name']
    # print(first_column_value)
    pp = image_preprocessing(df, '.')
    try:
        pp.augmentation('images-115-max-keys-400/'+first_column_value)
    except Exception as e:
        print("error is ",e, first_column_value)
        continue

# pp = image_preprocessing(df, '.')
# pp.augmentation('images-115-max-keys-400/images/AT_Tablet/AT_Tablet1.jpg')