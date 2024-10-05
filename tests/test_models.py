import pickle

import pytest
import json
import pandas as pd
from torchvision import transforms
import torch
import torch.utils.data as data
import numpy as np

from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROCESSED_TEST_IMAGES
from src.models.evaluate import load_image

@pytest.fixture
def best_model():
    """
    Returns the best model from the models folder. It is a fixture to be used in the test_best_model function
    """
    #open the scores.json file to get the name of the best model
    with open(METRICS_DIR / "scores.json", "r") as file:
        data_scores = json.load(file)
    best_model_name = data_scores["Run_name"] + ".pkl"

    #return the best model
    with open(MODELS_DIR / best_model_name, "rb") as f:
        return pickle.load(f)

@pytest.fixture
def cats_dogs_test_data():
    """
    Returns the test data to be used in the test_best_model function. It is a fixture to be used in the test_best_model function
    """
    def obtain_test_data(input_folder_path, input_test_images_path, batch_size):
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
    
    #Obtain the batch size from the scores.json file
    with open(METRICS_DIR / "scores.json", "r") as file:
        data_scores = json.load(file)
    batch_size = data_scores["Batch_size"]

    return obtain_test_data(PROCESSED_DATA_DIR, PROCESSED_TEST_IMAGES, batch_size)



def test_best_model(best_model,cats_dogs_test_data):
    """
    Function to test the best model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    best_model.eval()

    correct = 0
    total = 0 
    #print(cats_dogs_test_data)
    with torch.no_grad():
        for x, y in cats_dogs_test_data:
            x = x.to(device)  
            y = y.to(device)
            outputs = best_model(x)
            predicted = torch.argmax(outputs.logits, dim=1)
            y = y.squeeze()
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = (correct / total)

    assert accuracy == pytest.approx(1.0, rel=0.05)

