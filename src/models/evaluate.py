import json
import pickle
from pathlib import Path
from PIL import Image
from torchvision import transforms

import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from typing import Any

from src.config import METRICS_DIR, PROCESSED_DATA_DIR, PROCESSED_VALID_IMAGES, TRACKING_MLFLOW,EXPERIMENT_NAME

# Path to the models folder
MODELS_FOLDER_PATH = Path("models")




def load_image(image_path: str) -> Any:
    """
    Load an image from the specified path
    """

    image = Image.open(image_path).convert("RGB")  # Asegúrate de que la imagen esté en RGB
    return image




def preapare_validation_dataloaders(input_folder_path: Path, input_valid_images_path: Path,batch_size:int) -> data.DataLoader:
    """
    We get the validation dataloader
    """

    X_valid = pd.read_csv(input_folder_path / "X_valid.csv")
    y_valid = pd.read_csv(input_folder_path / "y_valid.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    validation_images = [transform(load_image(str(input_valid_images_path / f"image_valid_{i}.jpg"))) for i in range(len(X_valid))]
    valid_dataset = data.TensorDataset(torch.stack(validation_images), torch.tensor(y_valid.values).long())
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return valid_dataloader




def evaluate_model(model,batch_size: int) -> float:
    """
    Evaluate the model using the validation data.
    """

    valid_dataloader = preapare_validation_dataloaders(PROCESSED_DATA_DIR,PROCESSED_VALID_IMAGES,batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_dataloader):
            x = x.to(device)  # Mover los datos de entrada al dispositivo
            y = y.to(device)
            outputs = model(x)
            predicted = torch.argmax(outputs.logits, dim=1)
            y = y.squeeze()
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = (correct / total)*100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy







"""
Main function for the validation process
"""
#Load the parameters and run_ids from training script
with open("parameters_list.json", "r") as parameters_run_file:
    parameters_run = json.load(parameters_run_file)

with open("parameters_run.json", "r") as run_ids_file:
    run_ids = json.load(run_ids_file)

Path("metrics").mkdir(exist_ok=True)
metrics_folder_path = METRICS_DIR

#Set mlflow
mlflow.set_tracking_uri(TRACKING_MLFLOW)
mlflow.set_experiment(EXPERIMENT_NAME)

best_model = None
best_accuracy = 0

#Validation for each model
for combination in parameters_run:
    run_id = run_ids[combination]
    with mlflow.start_run(run_id=run_id) as run:
        run_name = combination + ".pkl"
        combination_params = parameters_run[combination]

        with open(MODELS_FOLDER_PATH / run_name, "rb") as pickled_model:
            model = pickle.load(pickled_model)

        accuracy = evaluate_model(model,combination_params["batch_size"])

        metrics_dict = {"Accuracy": accuracy}
        mlflow.log_metrics(metrics_dict)

    #To select the best model we focus on the accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = combination
        best_run_id = run.info.run_id

#Add the best model tag to the mlflow
with mlflow.start_run(run_id=best_run_id):  
        mlflow.set_tag("Best_model", "True")

#Save the metric of the best model in a json file
metrics_dict = {"Run_name":best_model,"Accuracy": best_accuracy}
with open(metrics_folder_path / "scores.json", "w") as scores_file:
    json.dump(
        metrics_dict,
        scores_file,
        indent=4,
    )

print("Evaluation completed.")
