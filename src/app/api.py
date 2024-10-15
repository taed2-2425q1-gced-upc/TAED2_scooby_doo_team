from http import HTTPStatus
from fastapi import FastAPI, UploadFile
from torchvision import transforms
from contextlib import asynccontextmanager
from PIL import Image
from src.config import MODELS_DIR
from typing import List
from torchsummary import summary
import torch
import io
import pickle
import numpy as np

model = None
image_counter = 0 

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
        model.eval()

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

        # Handle animated images (GIF, WebP) by extracting the first frame
        if hasattr(image, "is_animated") and image.is_animated:
            image.seek(0) 

        return torch.unsqueeze(transform(image),0)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file_format(file: UploadFile):
    return file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS

@app.post("/predict")
async def predict_image(files: List[UploadFile]):
    '''
    Classifies an image using one of our trained models.

    @param file: image to be classified, expected formats are  jpeg, png, gif and webp.
    @returns: classifiction of the image in the format {"class":"dog"} or {"class":"cat"}
    '''
    results = []
    try:
        for file in files:
            if not allowed_file_format(file):
                results.append({
                    "filename": file.filename,
                    "message": "Unsupported file format",
                    "status_code": HTTPStatus.BAD_REQUEST
                })
                continue

            image_bytes = await file.read()
            image = image_to_tensor(image_bytes)
            await file.close()

            prediction = np.argmax(model(image).logits.detach()[0]).tolist()

            if prediction == 0:
                prediction = "cat"
            else: 
                prediction = "dog"

            results.append ({
                "filename": file.filename,
                "message":HTTPStatus.OK.phrase,
                "status-code":HTTPStatus.OK,
                "class":prediction
            })
            image_counter += 1

        return results

    except Exception as e:
        response = {
            "message": HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
            "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "details": str(e)
        }
        return response

def get_model_summary():
    return summary(model, input_size=(3, 224, 224), verbose=0)

def get_model_parameters():
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@app.get("/model-stats")
async def model_stats():
    try:
        model_summary = get_model_summary() 
        model_params = get_model_parameters()  

        return {
            "message": "Model statistics",
            "status-code": HTTPStatus.OK,
            "trainable_parameters": model_params,
            "model_summary": str(model_summary)  
        }

    except Exception as e:
        return {
            "message": "Error fetching model stats",
            "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "details": str(e)
        }