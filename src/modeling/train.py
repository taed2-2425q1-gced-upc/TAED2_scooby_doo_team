import pickle
from pathlib import Path

import itertools
import mlflow
import random
import pandas as pd
import yaml
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch

from src.config import PROCESSED_DATA_DIR


#We set the experiment name if exists, if not, it will be created
mlflow.set_experiment("Dogs and cats classification")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

#Path of the parameters file
params_path = Path("params.yaml")

#Path of the prepared data folder
input_folder_path = PROCESSED_DATA_DIR

#TODO Read training dataset (data prepocessing should save the files in this folder) (DAVID TIENE QUE GUARDARLO EN ESE PATH)
X_train = pd.read_csv(input_folder_path / "X_train.csv")
y_train = pd.read_csv(input_folder_path / "y_train.csv")

#Read data preparation parameters
with open(params_path, "r", encoding="utf8") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["train"]
    except yaml.YAMLError as exc:
        print(exc)

#MODEL TRAINING AND EVALUATION
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
        #TODO TIPICO CODIGO DE TRAIN

