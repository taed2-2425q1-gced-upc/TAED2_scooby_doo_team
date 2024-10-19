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
import numpy as np
from datetime import timedelta
import zipfile
from typing import Union
import os
import json
from pathlib import Path
from datetime import datetime
from torchinfo import summary

image_counter = 0 
class_prediction_counts = {"cat": 0, "dog": 0,"unknown":0}
model_list = []
api_stats = Path("metrics/api_stats.json")
rating_models_api = Path("metrics/model_stats_api.json")
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



def save_ratings(ratings_dict):
    """
    Save the ratings to the ratings file
    """
    with open(rating_models_api, "w") as f:
        json.dump(ratings_dict, f, indent=4)



def load_api_stats():
    """
    Load the api stats from the api stats file
    """
    if api_stats.exists():
        with open(api_stats, "r") as f:
            return json.load(f)
    return {}



def save_api_stats(api_stats_dict):
    """
    Save the api stats to the api stats file
    """
    with open(api_stats, "w") as f:
        json.dump(api_stats_dict, f, indent=4)




ratings_data = load_ratings()
api_stats_data = load_api_stats()


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

    @param model_type: the model to use for classification
    @param file: image or zip to be classified, expected formats are  jpeg, png, gif and webp.
    @returns: classifiction of the image in the format {"class":"dog"} or {"class":"cat"}
    '''
    global image_counter
    global class_prediction_counts
    results = []

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
                        if model_name not in ratings_data:
                            ratings_data[model_name] = {}
                        if "image_processed" not in ratings_data[model_name]:
                            ratings_data[model_name]["image_processed"] = []
                        if "date_image_processed" not in ratings_data[model_name]:
                            ratings_data[model_name]["date_image_processed"] = []
                        ratings_data[model_name]["date_image_processed"].append(str(datetime.now()))
                        ratings_data[model_name]["image_processed"].append(1)
                        image_bytes = zip_file.read(zip_info.filename)
                        image_tensor = image_to_tensor(image_bytes)
                        
                        # Realizar la predicción para cada imagen dentro del ZIP
                        logits = model(image_tensor).logits.detach()[0]
                        probabilities = torch.softmax(logits, dim=0).numpy()
                        probabilities = [round(float(p), 3) for p in probabilities]
                        prediction_score = np.argmax(probabilities)

                        if probabilities[prediction_score] < 0.85:
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

                        image_counter += 1
                        save_ratings(ratings_data)

        else:
            if not allowed_file_format(file):
                results.append({
                    "filename": file.filename,
                    "message": "Unsupported file format",
                    "status_code": HTTPStatus.BAD_REQUEST
                })
                continue
            if model_name not in ratings_data:
                ratings_data[model_name] = {}
            if "image_processed" not in ratings_data[model_name]:
                ratings_data[model_name]["image_processed"] = []
            if "date_image_processed" not in ratings_data[model_name]:
                ratings_data[model_name]["date_image_processed"] = []
            ratings_data[model_name]["date_image_processed"].append(str(datetime.now()))
            ratings_data[model_name]["image_processed"].append(1)
            image_bytes = await file.read()
            image_tensor = image_to_tensor(image_bytes)
            await file.close()

            logits = model(image_tensor).logits.detach()[0]
            probabilities = torch.softmax(logits, dim=0).numpy()
            probabilities = [round(float(p), 3) for p in probabilities]
            prediction_score = np.argmax(probabilities)
            if probabilities[prediction_score] < 0.85:
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
            image_counter += 1
            save_ratings(ratings_data)

    return {"message": "Detailed predictions for all images", "results": results}




@app.get("/model-summary",tags=["Models"])
async def model_stats(Model_name: str):
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
    


@app.post("/models/{model_name}/rate", tags=["Rate models and API"])
def rate_model(model_name: str, rating: int):
    """
    Allow the user to rate a model from 1 to 5
    """
    if rating < 1 or rating > 5:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Rating must be between 1 and 5")
    
    #If the model is new, add it to the ratings data
    if model_name not in ratings_data:
        ratings_data[model_name] = {}
    if "ratings" not in ratings_data[model_name]:
        ratings_data[model_name]["ratings"] = []
    if "average_rating" not in ratings_data[model_name]:
        ratings_data[model_name]["average_rating"] = 0

    ratings_data[model_name]["ratings"].append(rating)
    ratings_data[model_name]["average_rating"] = round(sum(ratings_data[model_name]["ratings"]) / len(ratings_data[model_name]["ratings"]), 2)
    save_ratings(ratings_data)
    return {"message": "Rating added successfully"}



@app.get("/models/{model_name}/rating", tags=["Models"])
def get_model_rating(model_name: str):
    """
    Get the average rating of a specific model
    """
    if model_name not in ratings_data:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Model not found")

    return {
        "model_name": model_name,
        "average_rating": ratings_data[model_name]["average_rating"],
    }



@app.post("/rate_api", tags=["Rate models and API"])
def rate_api(rating: int):
    """
    Allow the user to rate the API from 1 to 5
    """
    if rating < 1 or rating > 5:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Rating must be between 1 and 5")

    if "api_ratings" not in api_stats_data:
        api_stats_data["api_ratings"] = {"ratings": [], "average_rating": 0}
    if "date" not in api_stats_data:
        api_stats_data["date"] = []

    api_stats_data["api_ratings"]["ratings"].append(rating)
    api_stats_data["api_ratings"]["average_rating"] = round(sum(api_stats_data["api_ratings"]["ratings"]) / len(api_stats_data["api_ratings"]["ratings"]), 2)
    api_stats_data["date"].append(str(datetime.now()))
    save_api_stats(api_stats_data)
    return {"message": "Rating added successfully"}





@app.get("/models_image_processed_vis", tags=["Visualization"])
def image_processed_per_model():
    """
    Endpoint to generate a line chart showing the number of images processed by each model per day for the current month.
    Each model will have its own line in the chart.
    """
    if not ratings_data:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="No data found for the models")
    
    current_month = datetime.now().month
    current_year = datetime.now().year

    last_day_of_month = (datetime(current_year, current_month + 1, 1) - timedelta(days=1)).day
    all_days_of_month = [datetime(current_year, current_month, day).date() for day in range(1, last_day_of_month + 1)]

    images_processed_per_model = {model_name: {day: 0 for day in all_days_of_month} for model_name in ratings_data.keys()}

    for model_name, model_data in ratings_data.items():
        if "date_image_processed" in model_data and "image_processed" in model_data:
            dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f').date() for date in model_data["date_image_processed"]]
            image_processed = model_data["image_processed"]

            # We filter the data to only include the current month and year
            current_month_dates = [date for date in dates if date.month == current_month and date.year == current_year]
            current_month_image_processed = [image_processed[i] for i in range(len(dates)) if dates[i].month == current_month and dates[i].year == current_year]

            for i, day in enumerate(current_month_dates):
                images_processed_per_model[model_name][day] += current_month_image_processed[i]

    plt.figure(figsize=(10, 5))

    for model_name, daily_data in images_processed_per_model.items():
        plt.plot(all_days_of_month, [daily_data.get(day, 0) for day in all_days_of_month], marker='o', linestyle='-', label=model_name)

    plt.title(f'Images Processed per Model for {datetime.now().strftime("%B %Y")}')
    plt.xlabel('Date')
    plt.ylabel('Images Processed')
    plt.legend(title="Models")
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")





@app.get("/api_rating_vis", tags=["Visualization"])
def api_rate_actual_month():
    """
    Endpoint to generate a line chart showing the average rating of the API for the current month
    """
    if "api_ratings" not in api_stats_data:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="No ratings found for the API")
    
    if "date" not in api_stats_data or "api_ratings" not in api_stats_data:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="No API rating data available")

    current_month = datetime.now().month
    current_year = datetime.now().year

    last_day_of_month = (datetime(current_year, current_month + 1, 1) - timedelta(days=1)).day
    all_days_of_month = [datetime(current_year, current_month, day).date() for day in range(1, last_day_of_month + 1)]

    dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f').date() for date in api_stats_data["date"]]
    ratings = api_stats_data["api_ratings"]["ratings"]

    if not dates or not ratings:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Insufficient data for plotting")

    # We filter the data to only include the current month and year
    current_month_dates = [date for date in dates if date.month == current_month and date.year == current_year]
    current_month_ratings = [ratings[i] for i in range(len(dates)) if dates[i].month == current_month and dates[i].year == current_year]

    if not current_month_dates or not current_month_ratings:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="No ratings for the current month")

    # We group the ratings by day
    daily_ratings = {}
    for i, day in enumerate(current_month_dates):
        if day not in daily_ratings:
            daily_ratings[day] = []
        daily_ratings[day].append(current_month_ratings[i])

    avg_rating_per_day = {day: np.mean(daily_ratings[day]) for day in daily_ratings}

    # We create a list of cumulative average ratings
    cumulative_average_ratings = []
    running_sum = 0
    count = 0
    for day in all_days_of_month:
        if day in avg_rating_per_day:
            running_sum += avg_rating_per_day[day]
            count += 1
        if count > 0:
            cumulative_average_ratings.append(running_sum / count)
        else:
            cumulative_average_ratings.append(0) 

    plt.figure(figsize=(10, 5))
    plt.plot(all_days_of_month, cumulative_average_ratings, marker='o', linestyle='-', color='b')
    plt.title(f'Cumulative Average API Rating for {datetime.now().strftime("%B %Y")}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Average Rating')
    plt.ylim(1, 5)
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")

