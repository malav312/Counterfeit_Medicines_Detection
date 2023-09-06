from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from identify_functions import identify
import pandas as pd
from verify import authentication

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files to serve CSS and JavaScript
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount static files to serve images
app.mount("/images", StaticFiles(directory="images"), name="images")

# Read CSV file containing medicine names
df = pd.read_csv("iqr_scores.csv")
medicine_names = df.iloc[:, 0].tolist()

@app.post("/authenticate")
def authenticate_user(image: UploadFile = File(...), medicine: str = Form(...)):
    # print("Here")
    image_path = f"images/{image.filename}"
    print("hERE",image_path)
    with open(image_path, "wb") as f:
        f.write(image.file.read())
    print("Here")
    # Call the authentication function
    result = authentication(image_path, medicine)
    
    return {"result": result}

@app.post("/identify")
def identify_medicine(image: UploadFile = File(...)):
    # Save the uploaded image
    image_path = f"images/{image.filename}"
    with open(image_path, "wb") as f:
        f.write(image.file.read())

    # Call the identification function
    result = identify(image_path)

    return {"result": result}

@app.get("/medicine_names")
def get_medicine_names():
    return {"medicine_names": medicine_names}

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "medicine_names": medicine_names})
