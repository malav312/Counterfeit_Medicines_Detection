import boto3
import os
from botocore.client import Config
import json
import traceback


class SpacesAPI:
    
    def __init__(self):        
        self.AWS_ACCESS_KEY_ID = 'DO00NZPZXLFD82FPKQEC'
        self.AWS_SECRET_ACCESS_KEY = 'nBgZH/wHNlXEgmaDpDjNAZEftoPhtXJBkc08wq8pJs0'
        self.AWS_REGION_NAME = 'sgp1' 
        self.BUCKET_NAME = 'malav312'
        self.ENDPOINT_URL = 'https://sgp1.digitaloceanspaces.com'

        session = boto3.session.Session()

        self.s3_client = boto3.client('s3',
                                endpoint_url=self.ENDPOINT_URL,
                                config=Config(s3={'addressing_style': 'virtual'}),
                                aws_access_key_id=self.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
                                region_name=self.AWS_REGION_NAME)

    def put_json(self,json_data:dict,object_key:str,metadata:dict=None,folder:str=None, bucket_name:str=None):
        """
        Args:
            json_data (dict):
            object_key (string):
            metadata (dict, optional):  Defaults to None.
            folder (string, optional):  Defaults to None.
            bucket_name (string, optional): Defaults to BUCKET_NAME.
            
        # Example usage: Upload a JSON object to S3 with public-read ACL
         sample_json = {
             "name": "Lenny",
             "age": 30,
             "website": "google.com",
             "address": {
                 "City": "New York",
                 "State": "NY",
                 "Street": "10th Avenue",
             }
         }
         s3_uploader = SpacesAPI()
         s3_uploader.put_json(json_data=sample_json,
                              object_key='sample.json',
                              metadata={"id": "54321", "type": "json"},
                              folder="json")
        """
        if(folder is None):
            key = object_key
        else:
            key = folder+'/'+object_key
            
        if(bucket_name is None):
            bucket_name = self.BUCKET_NAME

        if(metadata is None):
            metadata = {}
            
        print("Received request to upload object to S3", bucket_name,"/",key)
        
        self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=key,
                    Body=json.dumps(json_data).encode(),
                    ACL='public-read',
                    Metadata=metadata
                    )
        print(f'JSON uploaded to S3: {object_key}')
        
        
    def list_objects(self, bucket_name:str=None,max_keys:int=None,folder:str=None):
        """
        Args:
            bucket_name (string, optional): Defaults to BUCKET_NAME.
        """ 
        if(bucket_name is None):
            bucket_name = self.BUCKET_NAME
            
        if(max_keys is None and folder is None):
            response = self.s3_client.list_objects(Bucket=bucket_name)
            return response
            
        if(max_keys is not None and folder is None):
            response = self.s3_client.list_objects(Bucket=bucket_name,MaxKeys=max_keys)
            return response
        
        if(max_keys is None and folder is not None):
            response = self.s3_client.list_objects(Bucket=bucket_name,Prefix=folder)
            return response
        
        if(max_keys is not None and folder is not None):
            response = self.s3_client.list_objects(Bucket=bucket_name,Prefix=folder,MaxKeys=max_keys)
            return response
            
        response = self.s3_client.list_objects(Bucket=bucket_name)
        return response
    
    def get_object(self, object_key:str,folder:str=None, bucket_name:str=None):
        """_summary_

        Args:
            object_key (str): _description_
            bucket_name (str, optional): _description_. Defaults to None.
        """
        if(bucket_name is None):
            bucket_name = self.BUCKET_NAME
        
        if(folder is not None):
            object_key = folder+'/'+object_key
            
        response = self.s3_client.get_object(Bucket=bucket_name,Key=object_key)
        return response
    
    def get_object_json(self, object_key:str,folder:str=None, bucket_name:str=None):
        """_summary_

        Args:
            object_key (str): _description_
            bucket_name (str, optional): _description_. Defaults to None.
        """
        if(bucket_name is None):
            bucket_name = self.BUCKET_NAME
        
        if(folder is not None):
            object_key = folder+'/'+object_key
            
        try:
            response = self.s3_client.get_object(Bucket=bucket_name,Key=object_key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            print({"error":str(e),"traceback":traceback.format_exc(),"object_key":object_key})
            return None
        

# s3_helper = SpacesAPI()
# listobj = s3_helper.list_objects()
# # print(listobj['Key'] for i in listobj['Contents'])
# array_of_objects = [i['Key'] for i in listobj['Contents']]
# print("array of objects",array_of_objects)

# get_obj = s3_helper.get_object(object_key=array_of_objects[1])
# print(get_obj)
# print(get_obj['Body'].read().decode('utf-8'))

