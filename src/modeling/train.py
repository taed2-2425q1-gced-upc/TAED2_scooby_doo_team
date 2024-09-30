import pickle
import torch
import random
import itertools
import yaml
from PIL import Image
from torchvision import transforms

import pandas as pd
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from datetime import datetime

from pathlib import Path
from transformers import ViTForImageClassification
from src.config import PROCESSED_DATA_DIR,PROCESSED_TRAIN_IMAGES,MODELS_DIR
from typing import Any




def load_params_train(parameters_path: str) -> Any:
    """
    Loads the parameters from a YAML file
    """

    with open(parameters_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
            return params["train"]
        except yaml.YAMLError as exc:
            print(exc)
            return None




def combine_hyperparameters(values: tuple[tuple[Any]], number_of_combinations: int) -> Any:
    """
    Combines the provided hyperparameters and returns a random subsample of the specified size
    """

    combinations = list(itertools.product(*values))
    combinations = random.sample(combinations, number_of_combinations)
    return combinations 




def get_model(algorithm: str, model_name: str, targets: list[str]) -> Any:
    """
    Creates and returns a model based on the specified algorithm and model name
    """

    if algorithm == 'VisualTransformer':
        return ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=len(targets),
            ignore_mismatched_sizes=True
        )




def get_optimizer(algorithm: str, learning_rate: float, model: Any) -> Any:
    """
    Creates and returns an optimizer for the given model based on the specified algorithm
    """

    if algorithm == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif algorithm == 'sdg':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    



def prepare_hyperparameters_combinations(parameters: dict[str,Any]) -> tuple[dict[str,Any], list[str]]: 
    """
    Get the possible combination of hyperparameters and the names of those hyperparameters
    """   

    hyperparameter_names = parameters["hyperparameters"].keys()
    values = parameters['hyperparameters'].values()
    hyperparameter_combinations = combine_hyperparameters(values=values, number_of_combinations = parameters['combinations'])

    return  hyperparameter_combinations, hyperparameter_names




def load_image(image_path: str) -> Any:
    """
    Creates and returns a PIL image from the specified path
    """

    image = Image.open(image_path).convert("RGB")  # Asegúrate de que la imagen esté en RGB
    return image




def preapare_train_dataloaders(input_folder_path,input_train_images_path: Path,batch_size: int) -> data.DataLoader:
    """
    We get the training dataloaders
    """

    X_train = pd.read_csv(input_folder_path / "X_train.csv")
    y_train = pd.read_csv(input_folder_path / "y_train.csv")

    #Define the transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    training_images = [transform(load_image(str(input_train_images_path / f"image_train_{i}.jpg"))) for i in range(len(X_train))]
    train_dataset = data.TensorDataset(torch.stack(training_images), torch.tensor(y_train.values).long())
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader




def prepare_training_objects(targets,parameters_run: dict[str,Any]):
    """
    Prepares the training objects, including the model, optimizer, device, loss function, 
    and feature extractor based on the provided parameters
    """

    model = get_model(parameters_run['algorithm'],parameters_run['model_name'],targets)
    optimizer = get_optimizer(parameters_run['optimizer'],parameters_run['learning_rate'], model)
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.CrossEntropyLoss()

    return model, optimizer, device, loss_function




def train(parameters_run: dict[str, Any], model: Any, optimizer: Any, device: Any, loss_function: Any) -> None:
    """
    Trains a model for a specified number of epochs.
    """
    train_dataloader = preapare_train_dataloaders(PROCESSED_DATA_DIR,PROCESSED_TRAIN_IMAGES,parameters_run["batch_size"])

    model.to(device)
    model.train()
    loss_per_step = []
    loss_per_epoch = []
    for epoch in range(parameters_run["num_epochs"]):   
        print(f"Epoch {epoch+1}")    
        total_training_loss = 0
        for step, (x, y) in enumerate(train_dataloader):
            print(f"Step {step+1} and len of batch {len(x)}")
            optimizer.zero_grad()  
            x, y  = x.to(device), y.to(device)
            y = y.squeeze()
            output = model(x).logits
            loss = loss_function(output,y)
            loss.backward()       
            optimizer.step() 
            total_training_loss += loss.item()
            loss_per_step.append(loss.item())
        loss_per_epoch.append(sum(loss_per_step)/len(loss_per_step))
        print(f"Epoch {epoch+1}/{parameters_run['num_epochs']}, Average Training Loss: {total_training_loss / (step + 1)}") 

    return model,loss_per_epoch
        



def main_train():
    """
    Main function for the training process
    """

    params_path = Path("params.yaml")
    parameters = load_params_train(params_path)

    #Hyperparameters that we are going to use for the training
    hyperparameter_combinations, hyperparameter_names = prepare_hyperparameters_combinations(parameters)
    
    targets = parameters['targets']

    Path("models").mkdir(exist_ok=True)
    parameters_dict = {}
    metrics_dict = {}

    for combination in hyperparameter_combinations:
        parameters_run = dict(zip(hyperparameter_names,combination))
        model, optimizer, device, loss_function = prepare_training_objects(targets,parameters_run)
        model,loss_per_epoch = train(parameters_run, model, optimizer, device, loss_function)

        #Name of the model: algorithm_day_month_year_min
        now = datetime.now()
        run_id = f"day_{now.strftime('%d')}_month_{now.strftime('%m')}_year_{now.strftime('%Y')}_min_{now.strftime('%M')}"
        model_name = f"{parameters_run['algorithm']}"
        final_id = f"{model_name}_{run_id}"

        if final_id not in metrics_dict:
            metrics_dict[final_id] = loss_per_epoch

        if final_id not in parameters_dict:
            parameters_dict[final_id] = parameters_run

        #Save the model
        model_name = final_id + ".pkl"
        with open(MODELS_DIR / model_name, "wb") as pickle_file:
            pickle.dump(model, pickle_file)
    return parameters_dict,metrics_dict








    

