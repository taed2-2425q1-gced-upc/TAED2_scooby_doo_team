import pickle
from pathlib import Path

import itertools
import mlflow
import random
import pandas as pd
import yaml
from transformers import ViTForImageClassification, ViTFeatureExtractor,Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

from src.config import PROCESSED_DATA_DIR

#TODO DEJARLO BONITO CON FUNCIONES

#We set the experiment name if exists, if not, it will be created
mlflow.set_experiment("Dogs and cats classification")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

#Path of the parameters file
params_path = Path("params.yaml")

#Read data preparation parameters
with open(params_path, "r", encoding="utf8") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["train"]
    except yaml.YAMLError as exc:
        print(exc)

#Path of the prepared data folder
input_folder_path = PROCESSED_DATA_DIR

#TODO VER COMO SON  LOS DATOS Y ADAPTAR ESTA PARTE
X_train = pd.read_csv(input_folder_path / "X_train.csv")
y_train = pd.read_csv(input_folder_path / "y_train.csv")

x_train_tensor = data.TensorDataset(torch.tensor(X_train.values), torch.tensor(y_train.values))
y_train_tensor = data.TensorDataset(torch.tensor(X_train.values), torch.tensor(y_train.values))

train_loader = data.DataLoader(x_train_tensor, batch_size=params["batch_size"], shuffle=True,  num_workers=4)
test_loader  = data.DataLoader(y_train_tensor, batch_size=params["batch_size"], shuffle=False, num_workers=4)

#############################
#Model training and evaluation
#############################


keys = params["hyperparameters"].keys()
values = params["hyperparameters"].values()
combinations = list(itertools.product(*values))
random_combinations = random.sample(combinations, params["combinations"])

for comb in random_combinations:
    params = dict(zip(keys, comb))
    with mlflow.start_run():
        if params["algorithm"] == "VisualTransformer":
            feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                              num_labels=len(set(y_train)),
                                                              hidden_size=params["hidden_size"],
                                                              num_hidden_layers=params["num_hidden_layers"],
                                                              num_attention_heads=params["num_attention_heads"])
        if params["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_func = nn.CrossEntropyLoss()

        #Train the model
        for epoch in range(params["num_epochs"]):        
            for step, (x, y) in enumerate(train_loader):
                x = torch.tensor(np.stack(feature_extractor(x.numpy())['pixel_values'], axis=0)).float()
                x, y  = x.to(device), y.to(device)
                output = model(x)
                loss = loss_func(output,y)
                optimizer.zero_grad()  
                loss.backward()       
                optimizer.step() 
            print(f"Epoch {epoch+1}/{params['num_epochs']}, Loss: {loss.item()}") 
        
        #TODO Evaluate model