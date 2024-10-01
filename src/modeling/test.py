from pathlib import Path

import typer
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

from src.config import MODELS_DIR, PROCESSED_DATA_DIR,TRACKING_MLFLOW,EXPERIMENT_NAME,PROCESSED_TEST_IMAGES



def load_image(image_path: str) -> Any:
    """
    Load an image from the specified path
    """

    image = Image.open(image_path).convert("RGB")  # Asegúrate de que la imagen esté en RGB
    return image




def preapare_test_dataloaders(input_folder_path: Path, input_test_images_path: Path,batch_size:int) -> data.DataLoader:
    """
    We get the validation dataloader
    """

    X_test = pd.read_csv(input_folder_path / "X_test.csv")
    y_test = pd.read_csv(input_folder_path / "y_test.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_images = [transform(load_image(str(input_test_images_path / f"image_test_{i}.jpg"))) for i in range(len(X_test))]
    test_dataset = data.TensorDataset(torch.stack(test_images), torch.tensor(y_test.values).long())
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader

def get_best_model():
    """
    Get the best run and batch size of the best model from the experiment
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id)
    runs_df = pd.DataFrame([{
        **run.data.metrics,
        **run.data.params,
        "run_id": run.info.run_id
    } for run in runs])

    best_run = runs_df.loc[runs_df["Accuracy"].idxmax()]
    batch_size = int(best_run["batch_size"])
    run_name = best_run["name_model"]

    return run_name, batch_size


def test_model(model,batch_size: int) -> float:
    """
    Test the model and return the accuracy
    """
    test_dataloader = preapare_test_dataloaders(PROCESSED_DATA_DIR,PROCESSED_TEST_IMAGES,batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataloader):
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



def main_test():
    """
    Main function to test the model
    """

    mlflow.set_tracking_uri(TRACKING_MLFLOW)
    mlflow.set_experiment(EXPERIMENT_NAME)

    #Obtain the best run and the batch size of the best model
    run_name,batch_size = get_best_model()

    with open(MODELS_DIR / f"{run_name}.pkl", "rb") as f:
        model = pickle.load(f)

    #Test del modelo
    accuracy = test_model(model,batch_size)
    print(f"Accuracy in test: {accuracy:.2f}%")