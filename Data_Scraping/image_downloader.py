import json
import os
import requests

from spaces import SpacesAPI
from io import BytesIO
# from digitalocean import Spaces

# set up DigitalOcean Spaces client
import boto3
import base64
import sys

s3_client = SpacesAPI()

# define constants
json_dir = 'text'
img_dir = 'images'
limit_folders = 100
max_keys=400
url_to_skip = 'https://onemg.gumlet.io/l_watermark_346,w_240,h_240/a_ignore,w_240,h_240,c_fit,q_auto,f_auto/hx2gxivwmeoxxxsc1hix.png'
local_dir = '../data/output/images'


if(limit_folders <= len(os.listdir(local_dir))):
    print("Reached Limit of folders to download, exiting")
    sys.exit()

folders_completed_downloading=0

list_obj = s3_client.list_objects(folder=json_dir,max_keys=max_keys)

for i in list_obj['Contents']:
    obj_body = s3_client.get_object_json(i['Key'])
    if(obj_body['medicineURL'][0] == url_to_skip):
        
        if 'skip' in obj_body and obj_body['skip'] == True:
            print("skipping as already marked as skip")
            continue
        
        obj_body["skip"]=True
        s3_client.put_json(obj_body, i['Key'])
        print("Found a skip url, updating json obj")
        continue
    else:
        i['Key'] = i['Key'].replace('text','images')
        image_obj_data_json = s3_client.get_object_json(i['Key'])
        image_array = image_obj_data_json['medicineImage']
        print(len(image_array))
        medicineName = obj_body['medicineName']
        medicineName = medicineName.replace(" ","_")
        sub_folder = os.path.join(local_dir,medicineName)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
            print("made folder",sub_folder)
        if os.path.exists(sub_folder):
            print("folder exists",sub_folder)
            if len(os.listdir(sub_folder)) == len(image_array):
                print("skipping",medicineName,"as all images are already downloaded")
                continue
        
        count = 0
        for j in image_array:
            decoded_image_data = base64.b64decode(j)
            image_name = medicineName+str(count)+".jpg"
            count+=1
            with open(os.path.join(sub_folder,image_name),'wb') as f:
                f.write(decoded_image_data)
                f.close()
        print("saved image",image_name,"in folder",sub_folder+"/"+medicineName)
        folders_completed_downloading+=1
        print("---------completed downloading",folders_completed_downloading,"folders-------")
        if(folders_completed_downloading == limit_folders):
            print("completed downloading",limit_folders,"folders, exiting")
            break