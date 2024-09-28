import pickle
import mlflow
import torch
import random
import itertools
import yaml

import pandas as pd
import torch.nn as nn
import torch.utils.data as data
import numpy as np

from pathlib import Path
from transformers import ViTForImageClassification, ViTFeatureExtractor,Trainer, TrainingArguments
from src.config import PROCESSED_DATA_DIR
from typing import Any

def read_training_parameters(parameters_file: str) -> Any:
    # read the parameters file, section "train"
    with open(parameters_file, "r", encoding="utf8") as params_file:
        params = yaml.safe_load(params_file)
        params = params["train"]
        return params

def combine_hyperparameters(values: tuple[tuple[Any]], number_of_combinations: int) -> Any:
    # combine hyperparameters and select a random subsample
    combinations = list(itertools.product(*values))
    return random.sample(combinations, number_of_combinations) 

def get_model(algorithm: str, model_name: str, targets: list[str], hidden_size: int, number_hidden_layers: int, number_attention_heads) -> Any:
    if algorithm == 'VisualTransformer':
        return ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=len(targets),
            hidden_size=hidden_size,
            num_hidden_layers=number_hidden_layers,
            num_attention_heads=number_attention_heads
        )

def get_optimizer(algorithm: str, learning_rate: float, model: Any) -> Any:
    if algorithm == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif algorithm == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)

def get_feature_extractor(algorithm: str, model_name: str) -> Any:
    if algorithm == 'VisualTransformer':
        return ViTFeatureExtractor.from_pretrained(model_name)


def prepare_hyperparameters_combinations(parameters: dict[str,Any]) -> tuple[dict[str,Any], list[str]]:    
    # get the possible combination of hyperparameters and the names of those hyperparameters
    hyperparameter_names = parameters["hyperparameters"].keys()
    values = parameters['hyperparameters'].values()
    hyperparameter_combinations = combine_hyperparameters(values=values, number_of_combinations = parameters['combinations'])

    return  hyperparameter_combinations, hyperparameter_names

def preapare_dataloaders():
    # TODO: This funciton will change depending on the definitions of the datasets
    # get the training dataloader
    # get the test dataloader

    return ..., ...

def prepare_training_objects(parameters_run: dict[str,Any]):
    # create the model, optimizer, device, loss function and feature extractor
    model = get_model(parameters_run['algorithm'],parameters_run['model_name'],parameters_run['targets'],parameters_run["hidden_size"],parameters_run["num_hidden_layers"],parameters_run['num_attention_heads'])
    optimizer = get_optimizer(parameters_run['optimizer'],parameters_run['learning_rate'], model)
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.CrossEntropyLoss()
    feature_extractor = get_feature_extractor(parameters_run['algorithm'], parameters_run['model_name'])

    return model, optimizer, device, loss_function, feature_extractor


def train(parameters_run: dict[str, Any], train_dataloader: data.DataLoader, test_dataloader: data.DataLoader, model: Any, optimizer: Any, device: Any, loss_function: Any, feature_extractor: Any) -> None:
    for epoch in range(parameters_run["num_epochs"]):      
        model.train()  
        
        total_training_loss = 0
        for step, (x, y) in enumerate(train_dataloader):
            x = torch.tensor(np.stack(feature_extractor(x.numpy())['pixel_values'], axis=0)).float()
            x, y  = x.to(device), y.to(device)
            output = model(x)
            loss = loss_function(output,y)
            optimizer.zero_grad()  
            loss.backward()       
            optimizer.step() 
            total_training_loss += loss.item()
        print(f"Epoch {epoch+1}/{parameters_run['num_epochs']}, Average Training Loss: {total_training_loss / (step + 1)}") 

        model.eval()
        total_testing_loss = 0
        for step, (x,y) in enumerate(test_dataloader):
            output = model(x)
            loss = loss_function(output, y)
            total_testing_loss += loss.item()
        print(f"Epoch {epoch+1}/{parameters_run['num_epochs']}, Average Testing Loss: {total_training_loss / (step + 1)}") 
        


if __name__ == '__main__':
    #TODO read a configuration file or harcode the starting parameters, other than this the code is ready

    experiment_name = 'Dogs and cats classification'
    parameters = read_training_parameters('settings_path')

    hyperparameter_combinations, hyperparameter_names = prepare_hyperparameters_combinations(parameters)
    
    train_dataloader, test_dataloader = preapare_dataloaders()
    
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)
    for combination in hyperparameter_combinations:
        parameters_run = dict(zip(hyperparameter_names,combination))
        with mlflow.start_run():
            model, optimizer, device, loss_function, feature_extractor = prepare_training_objects(parameters_run)
            train(parameters_run, train_dataloader, test_dataloader, model, optimizer, device, loss_function, feature_extractor)

    

