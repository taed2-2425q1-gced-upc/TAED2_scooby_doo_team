"""
This script contains the test_best_model function that tests the best model.
"""

import pickle
import json
import pandas as pd
import torch
import pytest
from torchvision import transforms
from torch.utils import data

from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_TEST_IMAGES, PROCESSED_DATA_DIR
from src.models.evaluate import load_image

#@pytest.fixture
#def prepared_images_path():
#    return Path("/ruta/a/las/imagenes/procesadas")

@pytest.fixture
def x_train():
    return pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")

@pytest.fixture
def y_train():
    return pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")

@pytest.fixture
def x_valid():
    return pd.read_csv(PROCESSED_DATA_DIR / "X_valid.csv")

@pytest.fixture
def y_valid():
    return pd.read_csv(PROCESSED_DATA_DIR / "y_valid.csv")

@pytest.fixture
def x_test():
    return pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")

@pytest.fixture
def y_test():
    return pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")


@pytest.fixture
def best_model():
    """
    Returns the best model from the models folder.
    It is a fixture to be used in the test_best_model function
    """
    #open the scores.json file to get the name of the best model
    with open(METRICS_DIR / "scores.json", "r", encoding="utf-8") as file:
        data_scores = json.load(file)
    best_model_name = (
    data_scores["Run_name"] + ".pkl"
)

    #return the best model
    with open(MODELS_DIR / best_model_name, "rb") as f:
        return pickle.load(f)

@pytest.fixture
def cats_dogs_test_data():
    """
    Returns the test data to be used in the test_best_model function.
    It is a fixture to be used in the test_best_model function
    """
    def obtain_test_data(input_test_images_path, batch_size):

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        test_images = []
        test_labels = []

        class_folders = [folder for folder in input_test_images_path.iterdir() if folder.is_dir()]
        class_to_label = {folder.name: idx for idx, folder in enumerate(class_folders)}

        for class_folder in class_folders:
            label = class_to_label[class_folder.name]
            for image_path in class_folder.glob("*.jpg"):
                image = load_image(str(image_path))
                image = transform(image)
                test_images.append(image)
                test_labels.append(label)


        test_dataset = data.TensorDataset(torch.stack(test_images),
                                          torch.tensor(test_labels).long()
                                          )
        test_dataloader = data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False
                                          )

        return test_dataloader
    
    #Obtain the batch size from the scores.json file
    with open(METRICS_DIR / "scores.json", "r", encoding="utf-8") as file:
        data_scores = json.load(file)
    batch_size = data_scores["Batch_size"]

    return obtain_test_data(PROCESSED_TEST_IMAGES, batch_size)



def test_best_model(best_model,cats_dogs_test_data):
    """
    Function to test the best model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    best_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in cats_dogs_test_data:
            x = x.to(device)
            y = y.to(device)
            outputs = best_model(x)
            predicted = torch.argmax(outputs.logits, dim=1)
            y = y.squeeze()
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total

    assert accuracy == pytest.approx(0.95, rel=0.05)

'''
def test_cats_dogs_folders_have_images(prepared_images_path):
    """
    Verifies that the 'cats' and 'dogs' folders contain saved images.
    """
    cats_images = list((prepared_images_path / "cats").glob("*.jpg"))
    dogs_images = list((prepared_images_path / "dogs").glob("*.jpg"))
    
    assert len(cats_images) > 0
    assert len(dogs_images) > 0
'''


def test_train_has_more_images_than_valid_and_test(x_train, x_valid, x_test):
    """
    Checks that the training set has more images than the validation and test sets.
    """
    assert len(x_train) > len(x_valid)
    assert len(x_train) > len(x_test)


def test_data_csv_files_exist():
    """
    Checks if the data CSV files for train, valid, and test are saved correctly.
    """
    assert (PROCESSED_DATA_DIR / "X_train.csv").exists(), "X_train.csv file not found"
    assert (PROCESSED_DATA_DIR / "y_train.csv").exists(), "y_train.csv file not found"
    assert (PROCESSED_DATA_DIR / "X_valid.csv").exists(), "X_valid.csv file not found"
    assert (PROCESSED_DATA_DIR / "y_valid.csv").exists(), "y_valid.csv file not found"
    assert (PROCESSED_DATA_DIR / "X_test.csv").exists(), "X_test.csv file not found"
    assert (PROCESSED_DATA_DIR / "y_test.csv").exists(), "y_test.csv file not found"


def test_data_csv_row_counts(x_train, y_train, x_valid, y_valid, x_test, y_test):
    """
    Ensures that the number of rows in the CSV matches the number of images in each set.
    """
    assert len(pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")) == len(x_train), "Mismatch in X_train row count"
    assert len(pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")) == len(y_train), "Mismatch in y_train row count"
    assert len(pd.read_csv(PROCESSED_DATA_DIR / "X_valid.csv")) == len(x_valid), "Mismatch in X_valid row count"
    assert len(pd.read_csv(PROCESSED_DATA_DIR / "y_valid.csv")) == len(y_valid), "Mismatch in y_valid row count"
    assert len(pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")) == len(x_test), "Mismatch in X_test row count"
    assert len(pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")) == len(y_test), "Mismatch in y_test row count"