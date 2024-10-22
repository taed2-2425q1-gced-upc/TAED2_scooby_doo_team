'''
This module takes care of trainig some models, logging several metrics 
and dumping the models in a compressed format.
'''

import os
from typing import Any
from pathlib import Path
import pickle
import json
import random
import itertools
import mlflow

import pandas as pd
import torch
from torch import nn
from torch.utils import data

from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
from codecarbon import EmissionsTracker
from src.config import PROCESSED_TRAIN_IMAGES, \
    MODELS_DIR,METRICS_DIR,TRACKING_MLFLOW,EXPERIMENT_NAME
from src.features.prepare import load_params







def combine_hyperparameters(values: tuple[tuple[Any]], number_of_combinations: int) -> Any:
    """
    Combines the provided hyperparameters and returns a random subsample of the specified size
    """

    combinations = list(itertools.product(*values))
    combinations = random.sample(combinations, number_of_combinations)
    return combinations




def get_model(algorithm: str, model_name_get: str, targets_list: list[str]) -> Any:
    """
    Creates and returns a model based on the specified algorithm and model name
    """

    if algorithm == 'VisualTransformer':
        return ViTForImageClassification.from_pretrained(
            model_name_get,
            num_labels=len(targets_list),
            ignore_mismatched_sizes=True
        )
    return None




def get_optimizer(
        algorithm: str,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        model_opt: Any
    ) -> Any:
    """
    Creates and returns an optimizer for the given model based on the specified algorithm
    """

    if algorithm == 'adam':
        return torch.optim.Adam(model_opt.parameters(), lr=learning_rate,weight_decay=weight_decay)

    if algorithm == 'sgd':
        return torch.optim.SGD(
                model_opt.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum
            )

    return None



def prepare_hyperparameters_combinations(
        parameters_comb: dict[str,Any]
    ) -> tuple[dict[str,Any], list[str]]:
    """
    Get the possible combination of hyperparameters and the names of those hyperparameters
    """

    hyperparams_names = parameters_comb["hyperparameters"].keys()
    values = parameters_comb['hyperparameters'].values()
    hyperparameter_combs = combine_hyperparameters(
                                    values=values,
                                    number_of_combinations = parameters_comb['combinations']
                                )

    return  hyperparameter_combs, hyperparams_names




def load_image(image_path: str) -> Any:
    """
    Creates and returns a PIL image from the specified path
    """

    image = Image.open(image_path).convert("RGB")
    return image




def preapare_train_dataloaders(
        input_train_images_path: Path,
        batch_size: int
    ) -> data.DataLoader:
    """
    We get the training dataloaders
    """

    #Define the transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    training_images = []
    training_labels = []

    class_folders = [folder for folder in input_train_images_path.iterdir() if folder.is_dir()]
    class_to_label = {folder.name: idx for idx, folder in enumerate(class_folders)}

    for class_folder in class_folders:
        label = class_to_label[class_folder.name]  # Get the label for this class
        for image_path in class_folder.glob("*.jpg"):
            image = load_image(str(image_path))
            image = transform(image)  # Apply the transformations
            training_images.append(image)
            training_labels.append(label)

    train_dataset = data.TensorDataset(
                        torch.stack(training_images),
                        torch.tensor(training_labels).long()
                    )
    train_dataloader = data.DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True
                    )

    return train_dataloader




def prepare_training_objects(targets_train,parameters_run_train: dict[str,Any]):
    """
    Prepares the training objects, including the model, optimizer, device, loss function, 
    and feature extractor based on the provided parameters
    """

    training_model = get_model(parameters_run_train['algorithm'],
                               parameters_run_train['model_name'],targets_train)
    training_optimizer = get_optimizer(
                    parameters_run_train['optimizer'],
                    parameters_run_train['learning_rate'],
                    parameters_run_train['weight_decay'],
                    parameters_run_train['momentum'],
                    training_model
                )
    training_device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function_training = nn.CrossEntropyLoss()

    return training_model, training_optimizer, training_device, loss_function_training




def training(
        train_parameters_run: dict[str, Any],
        model_train: Any,
        optimizer_train: Any,
        device_train: Any,
        loss_function_train: Any
    ) -> None:
    """
    Trains a model for a specified number of epochs.
    """
    train_dataloader = preapare_train_dataloaders(
                        PROCESSED_TRAIN_IMAGES,
                        train_parameters_run["batch_size"]
                    )

    training_stats = {
        'loss_per_epoch': [],
        'loss_per_step': []
    }
    model_train.to(device_train)
    model_train.train()
    for epoch in range(train_parameters_run["num_epochs"]):
        print(f"Epoch {epoch+1}")
        epoch_loss = 0
        for step, (x, y) in enumerate(train_dataloader):
            print(f"Step {step+1} and len of batch {len(x)}")
            optimizer_train.zero_grad()
            x, y  = x.to(device_train), y.to(device_train).squeeze()

            output = model_train(x).logits
            loss = loss_function_train(output,y)
            loss.backward()
            optimizer_train.step()

            epoch_loss += loss.item()
            training_stats['loss_per_step'].append(loss.item())

            avg_loss = epoch_loss / (step + 1)
        training_stats['loss_per_epoch'].append(avg_loss)

        print(f"Epoch {epoch+1}/{train_parameters_run['num_epochs']}, \
             Average Training Loss: {avg_loss:.4f}")

    return model_train, training_stats['loss_per_epoch']


if __name__ == "__main__": # pragma: no cover
    rating_models_api = Path("metrics/model_stats_api.json")
    #We remove the file if it exists because we are going to create new models
    if rating_models_api.exists():
        os.remove(rating_models_api)

    params_path = Path("params.yaml")
    parameters = load_params(params_path,"train")
    emissions_output_folder = METRICS_DIR

    #Hyperparameters that we are going to use for the training
    hyperparameter_combinations, hyperparameter_names = (
        prepare_hyperparameters_combinations(parameters)
    )

    targets = parameters['targets']

    Path("models").mkdir(exist_ok=True)
    Path(emissions_output_folder).mkdir(parents=True, exist_ok=True)

    #logging.getLogger("codecarbon").disabled = True


    #Set mlflow
    mlflow.set_tracking_uri(TRACKING_MLFLOW)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except mlflow.exceptions.MlflowException as e:
        mlflow.create_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)

    parameters_dict = {}
    run_ids = {}

    for i,combination in enumerate(hyperparameter_combinations):
        parameters_run = dict(zip(hyperparameter_names,combination))

        final_id = f"Model_{i+1}"

        parameters_run["name_model"] = final_id

        with mlflow.start_run(run_name=final_id) as run:
            mlflow.log_params(parameters_run)

            if parameters_run["optimizer"] == "adam":
                parameters_run["momentum"] = 0
                parameters_run["weight_decay"] = 0

            model, optimizer, device, loss_function = (
                prepare_training_objects(targets,parameters_run)
            )


            #We are going to use the EmissionsTracker to track the emissions
            tracker = EmissionsTracker(
            project_name=EXPERIMENT_NAME,
            measure_power_secs=1,
            tracking_mode="process",
            output_dir=emissions_output_folder,
            output_file="emissions.csv",
            on_csv_write="append",
            default_cpu_power=45,
        )

            #si pasa que hay otra proceso corriendo es porque la version es la 2.6.0
            tracker.start()
            model,loss_per_epoch = training(parameters_run, model, optimizer, device, loss_function)
            tracker.stop()

            #We save the emissions metrics to mlflow
            emissions = pd.read_csv(emissions_output_folder / "emissions.csv")
            emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
            #emissions_params = emissions.iloc[-1, 13:].to_dict()
            #mlflow.log_params(emissions_params)
            mlflow.log_metrics(emissions_metrics)
            mlflow.set_tag("Duration units", "seconds")
            mlflow.set_tag("Emission Units", "kg CO2")
            mlflow.set_tag("Emission rate", "kg CO2/seg")
            mlflow.set_tag("Power units", "watts")
            mlflow.set_tag("Energy units", "kwh")


            for i in enumerate(loss_per_epoch):
                mlflow.log_metric("Train_loss", loss_per_epoch[i], step=i+1)

            if combination not in parameters_dict:
                parameters_dict[final_id] = parameters_run

            if combination not in run_ids:
                run_ids[final_id] = run.info.run_id

            #Save the model
            model_name = final_id + ".pkl"
            with open(MODELS_DIR / model_name, "wb") as pickle_file:
                pickle.dump(model, pickle_file)


    #Guarda parameters_list y run_ids en un json en la carpeta models
    with open("parameters_list.json", "w", encoding="utf-8") as parameters_file:
        json.dump(parameters_dict, parameters_file, indent=4)

    with open("parameters_run.json", "w", encoding="utf-8") as run_ids_file:
        json.dump(run_ids, run_ids_file, indent=4)

    print("Training finished")
