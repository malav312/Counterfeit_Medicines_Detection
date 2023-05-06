import json
import os
import requests

from io import BytesIO
# from digitalocean import Spaces

# set up DigitalOcean Spaces client
import boto3

# Create a Spaces client
spaces = boto3.client('s3',
                      region_name='sgp1',
                      endpoint_url='https://sgp1.digitaloceanspaces.com',
                      aws_access_key_id='DO00NZPZXLFD82FPKQEC',
                      aws_secret_access_key='nBgZH/wHNlXEgmaDpDjNAZEftoPhtXJBkc08wq8pJs0')

# define constants
json_dir = 'text'
img_dir = 'images'
limit_folders = 100
url_to_skip = "https://onemg.gumlet.io/l_watermark_346,w_240,h_240/a_ignore,w_240,h_240,c_fit,q_auto,f_auto/hx2gxivwmeoxxxsc1hix.png"
local_dir = '../data/output'

# get list of JSON files in the text folder
f = spaces.list_objects_v2(Bucket='malav312', Prefix=json_dir, MaxKeys='2') 
print(f) 
print(f['Contents'])
# print(json_files)
json_files = 'file'
for i, json_file in enumerate(json_files):
    if i >= limit_folders:
        break
    
    # download JSON file content
    json_str = spaces.get_object(Bucket='malav312', Key=json_file).content.decode('utf-8')
    json_data = json.loads(json_str)
    # check if the JSON file contains the URL to skip
    if json_data.get('image_url') == url_to_skip:
        print(f"Skipping JSON file {json_file.key} because it contains the URL to skip")
        continue

    # get corresponding image file from the images folder
    img_filename = os.path.basename(json_file.key).replace(' ', '_').replace('/', '-')
    img_file_path = os.path.join(img_dir, img_filename)

    if spaces.object_exists('malav312', img_file_path):
        # download image file content
        img_data = spaces.get_object('malav312', img_file_path).content

        # save image to local directory
        local_img_path = os.path.join(local_dir, img_filename)
        with open(local_img_path, 'wb') as f:
            f.write(img_data)

        print(f"Saved image file {img_file_path} to local directory {local_dir}")
    else:
        print(f"Image file {img_file_path} does not exist in the images folder")
