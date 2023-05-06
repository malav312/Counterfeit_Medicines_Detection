def getMedicineNames(filePath:str):
    '''
    input: (str) path to csv file
    output: [(str)] name of medicine
    '''
    import csv  
    medicineNames= []
    with open(filePath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        medicineNames = [row[1] for row in reader]
    return medicineNames


def getUrlForMedicine(medicine:str):
    '''
    input: (str) medicine name
    output: [(str)] list of strings, each being an image URL
    '''
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    # Define XPath to search box
    search_box_id = "srchBarShwInfo"

    # Set up headless browser
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    browser = webdriver.Chrome(options=options)

    # Load webpage
    browser.get("https://www.1mg.com/drugs/dolo-650-tablet-74467/")
    search_box_id = "srchBarShwInfo"

    search_box = browser.find_element(By.ID, search_box_id)
    search_box.send_keys(medicine)
    search_box.send_keys(Keys.RETURN)
    # Click on first search result
    search_results = browser.find_element(By.CLASS_NAME,'style__pro-title___3zxNC')
    search_results.click()

    browser.switch_to.window(browser.window_handles[0])
    browser.close()
    browser.switch_to.window(browser.window_handles[0])

    parent = browser.find_element(By.CLASS_NAME,"slick-track")
    children = parent.find_elements(By.TAG_NAME,"img")
    
    urlList = []
    for child in children:
        urlList.append(child.get_attribute("src"))

    browser.close()    
    browser.quit()
    return urlList


def getTextJSON(medicineName:str):
    '''
    input: (str) medicine name -> getUrlForMedicine()
    output: (JSON) where the keys are (str) medicineName  and [(str)] medicineURL 
    '''
    textData = {}
    textData["medicineName"] = medicineName
    
    medicineURL = getUrlForMedicine(medicineName) 
    textData["medicineURL"] = medicineURL

    return textData


def getImageJSON(medicineName:str,medicineURL:list):
    '''
    input: (str) medicine name and [(str)] medicineURL
    output: (JSON) storing binary blobs for image for every medicineURL 
    '''
    import requests
    import base64
    imageData = {}
    imageData["medicineName"] = medicineName
    
    images = []
    for url in medicineURL: 
        imageBinary = requests.get(url).content
        images.append(base64.b64encode(imageBinary).decode("utf-8"))
        # print(f"Done for {url} for {medicineName}")
    imageData["medicineImage"] = images

    return imageData


def writeJSON(jsonData:dict,outputLocation:str,medicineName:str):
    '''
    input: (JSON) data and (str) path to output file
    output: returns nothing, stores JSON to output location
    '''
    import json
    with open(outputLocation+medicineName+".json", 'w') as f:
        json.dump(jsonData, f)

def existence(medicineName:str,directory:str):
    '''
    input: (str) medicine name
    output: (bool) flag True if already exists
    '''
    import os
    return os.path.exists(f"../data/output/{directory}/{medicineName}.json")
    
def getDataForMedicine(medicineName:str,uploadToS3:bool = True):
    '''
    input: (str) medicine name
    output: returns nothing, gets textData and imageData and stores as JSON
    '''
    import json
    try: 
        if existence(medicineName,"text") is False:
            textData = getTextJSON(medicineName)
            writeJSON(textData,"../data/output/text/",medicineName)
        textData = json.load(open(f"../data/output/text/{medicineName}.json","r"))
        if existence(medicineName,"images") is False:
            imageData = getImageJSON(textData["medicineName"],textData["medicineURL"])
            writeJSON(imageData,"../data/output/images/",medicineName)
        imageData = json.load(open(f"../data/output/images/{medicineName}.json","r"))
        
        print(f"Completed for {medicineName}")
        if uploadToS3: 
            from spaces import SpacesAPI
            s3_uploader = SpacesAPI()
            s3_uploader.put_json(json_data=imageData,folder="images",object_key=f"{medicineName}.json")
            s3_uploader.put_json(json_data=textData,folder="text",object_key=f"{medicineName}.json",)
            print(f"Completed uploading to S3 for {medicineName}")
    except Exception as e: 
        print(f"Skipping for {medicineName}",e)


def displayImage(binaryData,isBase64Encoded:bool=False):
    '''
    input: (binary) image data
    output: returns nothing, displays image in human-friendly format
    '''
    from PIL import Image
    import io
    import base64
    if isBase64Encoded == True: 
        binaryData = base64.b64decode(binaryData)
    # Create an in-memory file object
    fileObject = io.BytesIO(binaryData)
    
    # Open the image file using Pillow
    image = Image.open(fileObject)
    
    # Display the image using Pillow
    image.show()

def vectorizeImage(binaryData,isBase64Encoded:bool=False):
    '''
    input: (binary) image data
    output: (npy) array with vector of size 512, using ResNet 18
    '''
    import torch
    from torch.autograd import Variable
    import torchvision.transforms as transforms
    import torchvision.models as models
    import torch.nn as nn
    from PIL import Image
    import base64
    
    if isBase64Encoded:
        binaryData = base64.b64decode(binaryData)
    img=binaryData
    # Load the pretrained model
    model = models.resnet18(pretrained=True)
    # Use the model object to select the desired layer
    layer = model._modules.get('avgpool')

    # Set model to evaluation mode
    model.eval()

    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # 1. Load the image with Pillow library
    # img = Image.open(imagePath)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer

    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(512))

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

def genAndSaveVectors():
    '''
    output: saves image vector as npy file
    '''
    import os 
    from PIL import Image
    import numpy as np 
    import warnings

    warnings.filterwarnings("ignore")

    dir_path = "../data/input/meds/images/"
    list_of_medicines = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    new_dir_path = "../data/output/vectors/"

    for medicine in list_of_medicines: 
        med_dir_path = dir_path+medicine+"/"
        list_of_photos = os.listdir(med_dir_path)
        for photo in list_of_photos:
            binaryImage = Image.open(med_dir_path+photo).convert("RGB")
            image_vector = np.array(vectorizeImage(binaryData=binaryImage))
            photo = photo.split(".")[0]
            if not os.path.exists(new_dir_path+f"{medicine}"):
                os.makedirs(new_dir_path+f"{medicine}")
            np.save(new_dir_path+f"{medicine}/{photo}.npy",image_vector)
            print(f"Done for {new_dir_path}{medicine}/{photo}")
            # arr = np.load(new_dir_path+f"{medicine}/{photo}.npy")
