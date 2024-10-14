from http import HTTPStatus
from fastapi import FastAPI, UploadFile
from torchvision import transforms
from contextlib import asynccontextmanager
from PIL import Image
from src.config import MODELS_DIR
import torch
import io
import pickle

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    Function for intialization and cleanup
    '''
    try:
        global model

        models_path = [
            filename
            for filename in MODELS_DIR.iterdir()
            if filename.suffix == ".pkl" 
        ]

        model_path = models_path[0]
        with open(model_path,"rb") as file:
            model = pickle.load(file)
        
        yield
    finally:
        del model

app = FastAPI(
    title="The ScoobyDoo project",
    description="This api allows you to use models for dogs vs cats classification",
    version="0.1",
    lifespan=lifespan,
)

def image_to_tensor(image_bytes: bytes):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),    # Resize to desired dimensions
            transforms.ToTensor(),            # Convert to tensor and normalize (0-255 to 0-1)
        ])

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        return transform(image)

@app.post("/predict")
async def predict_image(file: UploadFile):
    '''
    Classifies an image using one of our trained models.

    @param file: image to be classified, expected formats are  jpeg, png, gif and webp.
    @returns: classifiction of the image in the format {"class":"dog"} or {"class":"cat"}
    '''
    try:
        image_bytes = await file.read()
        image = image_to_tensor(image_bytes)
        await file.close()

        prediction = model(image)

        response = {
            "message":HTTPStatus.OK.phrase,
            "status-code":HTTPStatus.OK,
            "data":prediction
        }
        
        return response
    except:
        response = {
            "message": HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
            "status_code": HTTPStatus.INTERNAL_SERVER_ERROR
        }
        return response