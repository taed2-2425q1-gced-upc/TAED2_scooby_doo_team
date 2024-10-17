from http import HTTPStatus
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from torchvision import transforms
from contextlib import asynccontextmanager
from PIL import Image
from src.config import MODELS_DIR
from typing import List
import torch
import io
import pickle
import numpy as np
import zipfile
import os
from src.app.grad_cam import GradCAM 
import cv2
from torchinfo import summary

model = None
image_counter = 0 

@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    Function for intialization and cleanup. This will select the best model from the ones available on the server
    in order to make predictions.

    @param app: the api
    '''

    try:
        global model

        models_path = [
            filename
            for filename in MODELS_DIR.iterdir()
            if filename.suffix == ".pkl" 
        ]   
        if not models_path:
           raise FileNotFoundError("No model file found in the specified directory.")


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

def superimpose_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    # Resize the heatmap to match the input image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    # Convert the heatmap to RGB format
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file_format(file: UploadFile):
    return file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS

# In-memory dictionary for counts
class_prediction_counts = {"cat": 0, "dog": 0}

@app.post("/predict")
async def predict_image(files: List[UploadFile]):
    '''
    Classifies an image using one of our trained models.

    @param file: image to be classified, expected formats are  jpeg, png, gif and webp.
    @returns: classifiction of the image in the format {"class":"dog"} or {"class":"cat"}
    '''
    results = []
    # try:
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

        logits = model(image).logits.detach()[0]

        prediction = None
        prediction_score = np.argmax(logits).tolist()
        probabilities = torch.softmax(torch.tensor(logits), dim=0).numpy()

        if prediction_score == 0:
            prediction = "cat"
        else: 
            prediction = "dog"

        name, child = next(iter(model.named_children()))

        print(f'name {name}')
        print(f'chldred {child}')
        print(f'attention {child.encoder.layer[-1]}')

        class_prediction_counts[prediction] += 1
        grad_cam = GradCAM(model=model, target_layer=child.encoder.layer[-1])  # Adjust target layer based on your model
        heatmap = grad_cam(image, class_idx=prediction_score)

        # Convert the original image to numpy array for visualization
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Superimpose the heatmap onto the original image
        cam_image = superimpose_heatmap(image_np, heatmap)

        # Save the explanation (superimposed image)
        explanation_path = f"/tmp/{file.filename}_gradcam.png"
        Image.fromarray(cam_image).save(explanation_path)

        results.append({
            "filename": file.filename,
            "message":HTTPStatus.OK.phrase,
            "status-code":HTTPStatus.OK,
            "class":prediction,
            "probabilities": {
                "cat": probabilities[0],
                "dog": probabilities[1]
            },
            "prediction_counts": class_prediction_counts,
            "explanation_image_path": explanation_path
        })
        image_counter += 1

    return {"message": "Detailed predictions for all images", "results": results}

    # except Exception as e:
        # response = {
        #     "message": HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
        #     "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
        #     "details": str(e)
        # }
        # return response

def get_model_summary():
    """Return the summary of the model."""
    return summary(model, input_size=(1, 3, 224, 224), depth=4)


@app.get("/model-stats")
async def model_stats():
   global model
   if model is None:
       return {
           "message": "Model is not loaded.",
           "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
           "details": "Model is None."
       }
   try:
       model_summary = get_model_summary()
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


@app.get("/health")
async def health_check():
    try:
        # Check if model is loaded and ready
        model_status = "loaded" if model is not None else "not loaded"
        return {
            "status": "API is up and running!",
            "model_status": model_status
        }
    except Exception as e:
        return {"status": "API has issues", "error": str(e)}


@app.post("/evaluate-dataset")
async def evaluate_dataset(file: UploadFile):
   try:
       extract_to = "/tmp/dataset"
       os.makedirs(extract_to, exist_ok=True)


       contents = await file.read()
       zip_path = f"/tmp/{file.filename}"


       with open(zip_path, "wb") as buffer:
           buffer.write(contents)


       with zipfile.ZipFile(zip_path, 'r') as zip_ref:
           for file_info in zip_ref.infolist():
               if file_info.is_dir() or "__MACOSX" in file_info.filename or file_info.filename.startswith('.'):
                   continue
               zip_ref.extract(file_info, extract_to)


       # Process each image in the dataset folder
       predictions = []
       correct = 0
       total = 0
       has_labels = False


       for img_file in os.listdir(extract_to):
           img_path = os.path.join(extract_to, img_file)


           if os.path.isfile(img_path) and img_file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'webp')):
               with open(img_path, 'rb') as img_f:
                   image_bytes = img_f.read()
                   image = image_to_tensor(image_bytes)


               # Get model prediction
               logits = model(image).logits.detach()[0]
               prediction = np.argmax(logits).tolist()
               probabilities = torch.softmax(logits.clone().detach(), dim=0).numpy()


               prob_cat = float(probabilities[0])
               prob_dog = float(probabilities[1])




               # Assuming 0 is "cat" and 1 is "dog"
               class_name = "cat" if prediction == 0 else "dog"




               # Store prediction and probabilities
               predictions.append({
                   "filename": img_file,
                   "predicted_class": class_name,
                   "probabilities": {
                       "cat": prob_cat,
                       "dog": prob_dog
                   }
               })


               # Check if true labels exist in filenames (e.g., "cat1.jpg", "dog2.png")
               if "cat" in img_file or "dog" in img_file:
                   has_labels = True
                   true_label = "cat" if "cat" in img_file else "dog"
                   if class_name == true_label:
                       correct += 1
                   total += 1


       # Prepare response with or without accuracy depending on labels
       response = {
           "message": HTTPStatus.OK.phrase,
           "status-code": HTTPStatus.OK,
           "predictions": predictions,
           "total_images": len(predictions)
       }
       if has_labels and total > 0:
           accuracy = correct / total
           response["accuracy"] = accuracy
           response["labeled_images"] = total
        
       return response
  
   except Exception as e:
       return {
           "message": "Error processing ZIP file or images",
           "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
           "details": str(e)
       }
