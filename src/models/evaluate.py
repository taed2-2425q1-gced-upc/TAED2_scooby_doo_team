'''
This module takes care of validating models that we have generated
and assiging a tag to the best model found so far
'''

from pathlib import Path
from typing import Any
import json
import pickle
import mlflow
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from PIL import Image
from torchvision import transforms


from src.config import METRICS_DIR, PROCESSED_DATA_DIR, \
    PROCESSED_VALID_IMAGES, TRACKING_MLFLOW,EXPERIMENT_NAME

# Path to the models folder
MODELS_FOLDER_PATH = Path("models")




def load_image(image_path: str) -> Any:
    """
    Load an image from the specified path
    """

    image = Image.open(image_path).convert("RGB")  # Asegúrate de que la imagen esté en RGB
    return image




def preapare_validation_dataloaders(
        input_valid_images_path: Path,
        batch_size:int
    ) -> data.DataLoader:
    """
    We get the validation dataloader
    """


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    validation_images = []
    validation_labels = []

    class_folders = [folder for folder in input_valid_images_path.iterdir() if folder.is_dir()]
    class_to_label = {folder.name: idx for idx, folder in enumerate(class_folders)}

    for class_folder in class_folders:
        label = class_to_label[class_folder.name]  # Get the label for this class
        for image_path in class_folder.glob("*.jpg"):
            image = load_image(str(image_path))
            image = transform(image)  # Apply the transformations
            validation_images.append(image)
            validation_labels.append(label)

    valid_dataset = data.TensorDataset(
                        torch.stack(validation_images),
                        torch.tensor(validation_labels).long()
                    )
    valid_dataloader = data.DataLoader(
                        valid_dataset,
                        batch_size=batch_size,
                        shuffle=False
                    )

    return valid_dataloader




def evaluate_model(model,batch_size: int) -> float:
    """
    Evaluate the model using the validation data.
    """

    valid_dataloader = preapare_validation_dataloaders(
                        PROCESSED_VALID_IMAGES,
                        batch_size
                    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_dataloader):
            x = x.to(device)
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
if __name__ == "__main__": # pragma: no cover
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

    metric_to_save = {}

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
            info = {"params": combination_params, "metrics": accuracy}
            if combination not in metric_to_save:
                metric_to_save[combination] = {}
            metric_to_save[combination] = info

        #To select the best model we focus on the accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = combination
            best_run_id = run.info.run_id

    #Add the best model tag to the mlflow
    with mlflow.start_run(run_id=best_run_id):  
        mlflow.set_tag("Best_model", "True")

    with open(metrics_folder_path / "scores.json", "w") as scores_file:
        json.dump(
            metric_to_save,
            scores_file,
            indent=4,
        )

    print("Evaluation completed.")
