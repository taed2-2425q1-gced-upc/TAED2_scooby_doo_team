from http import HTTPStatus
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from torchvision import transforms
from contextlib import asynccontextmanager
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse
from PIL import Image
from src.config import MODELS_DIR
from typing import List
import torch
import io
import pickle
import csv
import numpy as np
from datetime import timedelta
import zipfile
from typing import Union
import os
import json
from pathlib import Path
from datetime import datetime
from torchinfo import summary

class_prediction_counts = {"cat": 0, "dog": 0,"unknown":0}
model_list = []
rating_models_api = Path("metrics/model_stats_api.json")
if not rating_models_api.exists():
    with open(rating_models_api, "w") as f:
        json.dump({}, f)

#CSVs
rating_api = Path("metrics/rating_api.csv")
models_procesed = Path("metrics/model_processed.csv")
models_rate = Path("metrics/model_rate.csv")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}




def allowed_file_format(file: UploadFile):
    """
    Check if the file format is allowed.
    """
    return file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS




def image_to_tensor(image_bytes: bytes):
    """
    Convert image bytes to a PyTorch tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),    # Resize to desired dimensions
        transforms.ToTensor(),            # Convert to tensor and normalize (0-255 to 0-1)
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Handle animated images (GIF, WebP) by extracting the first frame
    if hasattr(image, "is_animated") and image.is_animated:
        image.seek(0) 

    return torch.unsqueeze(transform(image),0)




def get_model_summary(model):
    """
    Return the summary of the model
    """
    return summary(model, input_size=(1, 3, 224, 224), depth=4)



def load_ratings():
    """
    Load the ratings from the ratings file
    """
    if rating_models_api.exists():
        with open(rating_models_api, "r") as f:
            return json.load(f)
    return {}


def save_ratings_json():
    """
    Save the ratings to the ratings file
    """
    with open(rating_models_api, "w") as f:
        json.dump(ratings_data, f, indent=4)




def save_rating_api_to_csv(rating: int):
    """
    Guardar la calificación en el archivo rating_api.csv, junto con el día, mes y año.
    """
    file_exists = rating_api.exists()

    with open(rating_api, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(["Rating", "Day", "Month", "Year"])
        
        current_time = datetime.now()
        day = current_time.day
        month = current_time.month
        year = current_time.year
        
        writer.writerow([rating, day, month, year])



def save_rating_models_to_csv(model_name,rating: int):
    """
    Save the rating in the file rating_models.csv, along with the day, month and year.
    """
    file_exists = models_rate.exists()

    with open(models_rate, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(["Model_name","Rating", "Day", "Month", "Year"])
        
        current_time = datetime.now()
        day = current_time.day
        month = current_time.month
        year = current_time.year
        
        writer.writerow([model_name,rating, day, month, year])



def save_processed_images_to_csv(model_name, count: int):
    """
    Save the number of processed images in the file model_processed.csv, along with the day, month and year.
    """
    file_exists = models_procesed.exists()

    with open(models_procesed, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(["Model_name","Processed_images", "Day", "Month", "Year"])
        
        current_time = datetime.now()
        day = current_time.day
        month = current_time.month
        year = current_time.year
        
        writer.writerow([model_name,count, day, month, year])


ratings_data = load_ratings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    Loads all pickled models found in `MODELS_DIR` and adds them to `models_list`
    @param app: the api
    '''
    global model_list
    model_paths = [
        filename
        for filename in MODELS_DIR.iterdir()
    ]
    for path in model_paths:
        try:
            with open(path, "rb") as file:
                model_wrapper = pickle.load(file)
                model_list.append(model_wrapper)
        except (pickle.UnpicklingError, FileNotFoundError, IsADirectoryError) as e:
            print(f"Error loading model from {path}: {e}")
    yield
    del model_list




app = FastAPI(
    title="The ScoobyDoo project",
    description="This api allows you to use models for dogs vs cats classification",
    version="0.1",
    lifespan=lifespan,
)


@app.get("/")
async def read_root():
    """Root endpoint"""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to ScoobyDoo project API!"},
    }
    return response




@app.get("/models_available", tags=["Models"])
def _get_models_list():
    """
    Return the list of available models
    """
    global model_list
    available_models = []
    with open("metrics/scores.json", "r") as scores_file:
        scores_dict = json.load(scores_file)

    for model_name in scores_dict.keys():
        accuracy = scores_dict[model_name]["metrics"]
        model_dict = {
            "name": model_name,
            "validation_accuracy": accuracy,
        }
        available_models.append(model_dict)

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": available_models,
    }




@app.post("/predict",tags=["Prediction"])
async def predict_image(model_name: str,files: List[UploadFile]):
    '''
    Classifies an image using one of our trained models.

    @param model_type: the model to use for classification. The structure of the model name is "Model_<model_number>"
    @param file: image or zip to be classified, expected formats are  jpeg, png, gif and webp.
    @returns: classifiction of the image in the format {"class":"dog"} or {"class":"cat"}
    '''
    global class_prediction_counts
    results = []
    count = 0

    model_idx = int(model_name.split("_")[-1]) - 1
    if model_idx >= len(model_list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Model not found. Please check the model name and try again.",
        )
    model = model_list[model_idx]
    for file in files:
        if file.filename.endswith('.zip'):
            # Extraer el contenido del ZIP
            with zipfile.ZipFile(io.BytesIO(await file.read())) as zip_file:
                for zip_info in zip_file.infolist():
                    if zip_info.filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        count += 1
                        image_bytes = zip_file.read(zip_info.filename)
                        image_tensor = image_to_tensor(image_bytes)
                        
                        # Realizar la predicción para cada imagen dentro del ZIP
                        logits = model(image_tensor).logits.detach()[0]
                        probabilities = torch.softmax(logits, dim=0).numpy()
                        probabilities = [round(float(p), 3) for p in probabilities]
                        prediction_score = np.argmax(probabilities)

                        if probabilities[prediction_score] < 0.58:
                            prediction = "unknown"
                        else:
                            if prediction_score == 0:
                                prediction = "cat"
                            else: 
                                prediction = "dog"

                        class_prediction_counts[prediction] += 1

                        # Añadir los resultados para cada imagen en el ZIP
                        results.append({
                            "filename": zip_info.filename,
                            "message": HTTPStatus.OK.phrase,
                            "status-code": HTTPStatus.OK,
                            "class": prediction,
                            "probabilities": {
                                "cat": probabilities[0],
                                "dog": probabilities[1]
                            },
                            "prediction_counts": class_prediction_counts
                        })


        else:
            if not allowed_file_format(file):
                results.append({
                    "filename": file.filename,
                    "message": "Unsupported file format",
                    "status_code": HTTPStatus.BAD_REQUEST
                })
                continue
            count += 1
            image_bytes = await file.read()
            image_tensor = image_to_tensor(image_bytes)
            await file.close()

            logits = model(image_tensor).logits.detach()[0]
            probabilities = torch.softmax(logits, dim=0).numpy()
            probabilities = [round(float(p), 3) for p in probabilities]
            prediction_score = np.argmax(probabilities)
            if probabilities[prediction_score] < 0.58:
                prediction = "unknown"
            else:
                if prediction_score == 0:
                    prediction = "cat"
                else: 
                    prediction = "dog"

            class_prediction_counts[prediction] += 1

            results.append({
                "filename": file.filename,
                "message":HTTPStatus.OK.phrase,
                "status-code":HTTPStatus.OK,
                "class":prediction,
                "probabilities": {
                    "cat": probabilities[0],
                    "dog": probabilities[1]
                },
                "prediction_counts": class_prediction_counts
            })
    save_processed_images_to_csv(model_name,count)
    return {"message": "Detailed predictions for all images", "results": results}




@app.get("/model-summary",tags=["Models"])
async def model_stats(Model_name: str):
   """
   Get the summary of the model

    @param Model_name: the name of the model to get the summary. The structure of the model name is "Model_<model_number>"
    @returns: the summary of the model
   """
   global model_list
   
   model_idx = int(Model_name.split("_")[-1]) - 1
   if model_idx >= len(model_list):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Model not found. Please check the model name and try again.",
            ) 
   model = model_list[model_idx]
   try:
       model_summary = get_model_summary(model)
       summary_str = str(model_summary)
       status_code = HTTPStatus.OK


       return HTMLResponse(
           content=f"""{summary_str},
           Status Code: {status_code}
           """
       )


   except Exception as e:
       return {
           "message": "Error fetching model stats",
           "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
           "details": str(e)
       }




@app.get("/health",tags=["Models"])
async def health_check(Model_name: str):
    """
    Check the health of the API and the model

    @param Model_name: the name of the model to check. The structure of the model name is "Model_<model_number>"
    @returns: the status of the API and the model
    """
    global model_list
    model_idx = int(Model_name.split("_")[-1]) - 1
    if model_idx >= len(model_list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Model not found. Please check the model name and try again.",
        )
    model = model_list[model_idx]

    try:
        # Check if model is loaded and ready
        model_status = "loaded" if model is not None else "not loaded"
        return {
            "status": "API is up and running!",
            "model_status": model_status
        }
    except Exception as e:
        return {"status": "API has issues", "error": str(e)}
    


@app.post("/models/rate", tags=["Rate models and API"])
def rate_model(model_name: str, rating: int):
    """
    Allow the user to rate a model from 1 to 5.

    @param model_name: the name of the model to rate. The structure of the model name is "Model_<model_number>"
    @param rating: the rating to assign to the model. It must be an integer between 1 and 5
    @returns: a message indicating if the rating was added successfully
    """
    if rating < 1 or rating > 5:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Rating must be between 1 and 5")
    idx = int(model_name.split("_")[-1]) - 1
    if idx >= len(model_list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Model not found. Please check the model name and try again.",
        )
    
    if model_name not in ratings_data:
        ratings_data[model_name] = {}
    if "ratings" not in ratings_data[model_name]:
        ratings_data[model_name]["ratings"] = []
    if "average_rating" not in ratings_data[model_name]:
        ratings_data[model_name]["average_rating"] = 0
    print(ratings_data)
    ratings_data[model_name]["ratings"].append(rating)
    if len(ratings_data[model_name]["ratings"]) > 30:
        ratings_data[model_name]["ratings"].pop(0)
    ratings_data[model_name]["average_rating"] = round(sum(ratings_data[model_name]["ratings"]) / len(ratings_data[model_name]["ratings"]), 2)

    save_ratings_json()
    save_rating_models_to_csv(model_name,rating)

    return {"message": "Rating added successfully"}



@app.get("/models/rating", tags=["Models"])
def get_model_rating(model_name: str):
    """
    Get the average rating of a specific model

    @param model_name: the name of the model to get the rating. The structure of the model name is "Model_<model_number>"
    @returns: the average rating of the model of the last 30 ratings
    """
    idx = int(model_name.split("_")[-1]) - 1
    if idx >= len(model_list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Model not found. Please check the model name and try again.",
        )
    if model_name not in ratings_data:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Model not rated yet")

    return {
        "model_name": model_name,
        "average_rating": ratings_data[model_name]["average_rating"],
    }



@app.post("/rate_api", tags=["Rate models and API"])
def rate_api(rating: int):
    """
    Allows the user to rate the API from 1 to 5.

    @param rating: la calificación a asignar a la API. Debe ser un entero entre 1 y 5.
    """

    if rating < 1 or rating > 5:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Rating must be between 1 and 5")
    save_rating_api_to_csv(rating)

    return {"message": "Rating added successfully"}
