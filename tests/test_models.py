"""
This script contains the test_best_model function that tests the best model.
"""

import pickle
import json
from pathlib import Path

import torch
import pytest
from torchvision import transforms
from torch.utils import data
from PIL import Image
import yaml
from unittest.mock import patch, mock_open

from src.config import PROCESSED_DATA_DIR,RAW_DATA_DIR,METRICS_DIR, MODELS_DIR, PROCESSED_TRAIN_IMAGES,PROCESSED_TEST_IMAGES,PROCESSED_VALID_IMAGES,MODELS_DIR
from src.features.prepare import load_data,load_params,split_data,save_data
from src.models.train import combine_hyperparameters,prepare_training_objects,get_model,prepare_hyperparameters_combinations,get_optimizer,preapare_train_dataloaders,training
from src.models.evaluate import preapare_validation_dataloaders,load_image,evaluate_model


#Prepare fixtures

@pytest.fixture
def loaded_data():
    """
    Fixture to load the data once and reuse it in multiple tests
    """
    df = load_data(RAW_DATA_DIR)
    return df




@pytest.fixture
def mock_prepare_parameters():
    """
    Returns a dictionary with the parameters for the preprocessing
    """
    return {
        "train_size": 0.01,
        "test_size": 0.025,
        "valid_size": 0.7,
        "random_state": 2024
    }




@pytest.fixture
def split_data_mock(loaded_data,mock_prepare_parameters):
    """
    Fixture to split the data for the tests.
    """
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(loaded_data, mock_prepare_parameters)
    return x_train, x_valid, x_test, y_train, y_valid, y_test




@pytest.fixture
def mock_yaml_file(tmp_path):
    """
    Returns the path to a mock yaml file
    """

    yaml_content = """
    prepare:
        train_size: 0.6
        test_size: 0.5
        valid_size: 0.5
        random_state: 2024
    train:
        algorithm: VisualTransformer
        model_name: google/vit-base-patch16-224
        batch_size: 32
        num_epochs: 10
        learning_rate: 0.001
        optimizer: adam
        weight_decay: 0.0001
        momentum: 0.9
    """
    yaml_path = tmp_path / "params_test.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path




#Train, validation fixtures

@pytest.fixture
def sample_image(tmp_path):
    """
    Returns the path to a sample image
    """
    # Crear una imagen temporal para usar en el test
    img_path = "data/processed/valid_images/dogs/image_valid_0.jpg"
    return img_path




@pytest.fixture
def mock_train_parameters():
    """
    Returns a dictionary with the parameters for the training
    """

    return {
        "batch_size": 32,
        "num_epochs": 2,
        "algorithm": "VisualTransformer",
        "model_name": "google/vit-base-patch16-224",
        "learning_rate": 0.001,
        "optimizer": "adam",
        "weight_decay": 0.0001,
        "momentum": 0.9
    }

#Test fixtures

@pytest.fixture
def best_model():
    """
    Returns the best model from the models folder.
    It is a fixture to be used in the test_best_model function
    """
    #open the scores.json file to get the name of the best model
    with open(METRICS_DIR / "scores.json", "r", encoding="utf-8") as file:
        data_scores = json.load(file)
    #recorrer los modelos y seleccionar la key con el mejor score
    best_model_name = None
    best_score = 0
    for model_name, value in data_scores.items():
        if value["metrics"] > best_score:
            best_score = value["metrics"]
            best_model_name = model_name
    best_model_name = best_model_name + ".pkl"
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
    best_model_name = None
    best_score = 0
    for model_name, value in data_scores.items():
        if value["metrics"] > best_score:
            best_score = value["metrics"]
            best_model_name = model_name
    batch_size = data_scores[best_model_name]["params"]["batch_size"]

    return obtain_test_data(PROCESSED_TEST_IMAGES, batch_size)


#PREPARE SCRIPT TESTS

def test_load_data(loaded_data):
    """
    Function to test the load_data function
    """
    
    assert not loaded_data.empty, "Dataframe is empty"
    assert 'image' in loaded_data.columns, "The column 'image' is not present"
    assert 'labels' in loaded_data.columns, "The column 'labels' is not present"




def test_load_params_prepare(mock_yaml_file):
    """
    Function to test the load_params_prepare function
    """

    params = load_params(mock_yaml_file,"prepare")
    assert params is not None, "Parameters are not loaded correctly"
    assert params["valid_size"] == 0.5, "Valid size is not correct"
    assert params["random_state"] == 2024, "Random state is not correct"
    assert params["train_size"] == 0.6, "Train size is not correct"
    assert params["test_size"] == 0.5, "Test size is not correct"




def test_load_params_yaml_error():
    """
    Function to test that an exception in load_params is handled correctly
    """
    with patch("builtins.open", mock_open(read_data="invalid data")):
        with patch("yaml.safe_load", side_effect=yaml.YAMLError("Mock YAML error")):
            params = load_params("fake_path.yaml", "prepare")
            assert params is None, "YAMLError was not handled correctly"




def test_split_data(loaded_data,mock_prepare_parameters):
    """
    Function to test the split_data function
    """

    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(loaded_data, mock_prepare_parameters)
    assert len(x_train) > 0, "x_train is empty"
    assert len(x_valid) > 0, "x_valid is empty"
    assert len(x_test) > 0, "x_test is empty"
    assert len(y_train) == len(x_train), "Mismatch between x_train and y_train"
    assert len(y_valid) == len(x_valid), "Mismatch between x_valid and y_valid"
    assert len(y_test) == len(x_test), "Mismatch between x_test and y_test"
    total_data_size = len(loaded_data)
    
    expected_train_size = int(total_data_size * mock_prepare_parameters['train_size'])
    expected_valid_test = int(total_data_size * mock_prepare_parameters['test_size'])
    expected_valid_size = int(expected_valid_test* mock_prepare_parameters['valid_size'])
    expected_test_size = int(expected_valid_test - expected_valid_size)

    assert len(x_train) == expected_train_size or len(x_train) == expected_train_size+1, f"Train set size is incorrect, expected {expected_train_size}, got {len(x_train)}"
    assert len(x_valid) == expected_valid_size or len(x_valid) == expected_valid_size+1, f"Validation set size is incorrect, expected {expected_valid_size}, got {len(x_valid)}"
    assert len(x_test) == expected_test_size or len(x_test) == expected_test_size+1 , f"Test set size is incorrect, expected {expected_test_size}, got {len(x_test)}"




def test_save_data(split_data_mock):

    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data_mock
    save_data(x_train, y_train, x_valid, y_valid, x_test, y_test)
    assert (PROCESSED_DATA_DIR / "X_train.csv").exists(), "File X_train.csv does not exist"
    assert (PROCESSED_DATA_DIR / "y_train.csv").exists(), "File y_train.csv does not exist"
    assert (PROCESSED_DATA_DIR / "X_valid.csv").exists(), "File X_valid.csv does not exist"
    assert (PROCESSED_DATA_DIR / "y_valid.csv").exists(), "File y_valid.csv does not exist"
    assert (PROCESSED_DATA_DIR / "X_test.csv").exists(), "File X_test.csv does not exist"
    assert (PROCESSED_DATA_DIR / "y_test.csv").exists(), "File y_test.csv does not exist"

    cats_train_dir = PROCESSED_TRAIN_IMAGES / "cats"
    dogs_train_dir = PROCESSED_TRAIN_IMAGES / "dogs"
    cats_valid_dir = PROCESSED_VALID_IMAGES / "cats"
    dogs_valid_dir = PROCESSED_VALID_IMAGES / "dogs"
    cats_test_dir = PROCESSED_TEST_IMAGES / "cats"
    dogs_test_dir = PROCESSED_TEST_IMAGES / "dogs"

    assert cats_train_dir.exists(), "Directory cats does not exist in train images"
    assert dogs_train_dir.exists(), "Directory dogs does not exist in train images"
    assert cats_valid_dir.exists(), "Directory cats does not exist in valid images"
    assert dogs_valid_dir.exists(), "Directory dogs does not exist in valid images"
    assert cats_test_dir.exists(), "Directory cats does not exist in test images"
    assert dogs_test_dir.exists(), "Directory dogs does not exist in test images"
    assert len(list(cats_train_dir.glob("*.jpg"))) > 0, "No cat images in train images"
    assert len(list(dogs_train_dir.glob("*.jpg"))) > 0, "No dog images in train images"
    assert len(list(cats_valid_dir.glob("*.jpg"))) > 0, "No cat images in valid images"
    assert len(list(dogs_valid_dir.glob("*.jpg"))) > 0, "No dog images in valid images"
    assert len(list(cats_test_dir.glob("*.jpg"))) > 0, "No cat images in test images"
    assert len(list(dogs_test_dir.glob("*.jpg"))) > 0, "No dog images in test images"


#TRAINING SCRIPT TESTS

def test_preapare_train_dataloaders():
    """
    Function to test that the DataLoader is correctly generated
    """

    batch_size = 32
    train_dataloader = preapare_train_dataloaders(
                        PROCESSED_TRAIN_IMAGES,
                        batch_size
                    )
    
    batch = next(iter(train_dataloader))
    print(batch[0].shape)
    assert len(batch) == 2, "DataLoader has not the expected number of elements"
    assert batch[0].shape[0] == batch_size, "The batch size is not correct"
    assert batch[0].shape[2] == 224, "The height of the image is not correct"
    assert batch[0].shape[3] == 224, "The width of the image is not correct"
    assert batch[0].shape[1] == 3, "The number of channels is not correct"
    assert len(train_dataloader) > 0, "Dataloader does not have any data"




def test_get_combine_hyperparameters():
    """
    Function to test that the hyperparameters are correctly combined
    """

    hyperparameters = (('adam', 'sgd'), (0.001, 0.01), (32, 64),(0.95,0.99))
    num_combinations = 4
    combinations = combine_hyperparameters(hyperparameters, num_combinations)

    assert len(combinations) == num_combinations, "Combinations have not the expected length."
    for combo in combinations:
        assert combo[0] in ('adam', 'sgd'), "Optimizer is not correct"
        assert combo[1] in (0.001, 0.01), "Learning rate is not correct"
        assert combo[2] in (32, 64), "Batch size is not correct"
        assert combo[3] in (0.95, 0.99), "Momentum is not correct"




def test_prepare_hyperparameters_combinations():
    """
    Function to test that the hyperparameters are correctly prepared
    """
    parameters = {
        "hyperparameters": {
            "learning_rate": [0.01, 0.001],
            "batch_size": [16, 32]
        },
        "combinations": 4
    }

    combinations, names = prepare_hyperparameters_combinations(parameters)
    
    assert combinations is not None
    assert list(names) == ["learning_rate", "batch_size"]




def test_get_model():
    """
    Function to test that in get_model funtion, the model is correctly initialized
    """

    model = get_model("VisualTransformer", "google/vit-base-patch16-224", ["cat", "dog"])
    assert model is not None, "The model has not been initialized correctly"
    assert model.config.num_labels == 2, "Number of labels is not correct."




def test_get_optimizer(best_model):
    """
    Function to test that in get_optimizer funtion, the optimizer is correctly initialized
    """

    #Example of parameters for th"""e optimizer if the optimizer is adam
    optimizer = get_optimizer("adam", 0.001, 0.0001, 0.9, best_model)
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer has not the expected type"
    assert optimizer.defaults['lr'] == 0.001, "Learning rate is not correct"
    assert optimizer.defaults['weight_decay'] == 0.0001, "Weight decay is not correct"
    assert 'momentum' not in optimizer.defaults, "Momentum is not correct"
    #Example of parameters for the optimizer if the optimizer is sgd
    optimizer = get_optimizer("sgd", 0.001, 0.0001, 0.9, best_model)
    assert isinstance(optimizer, torch.optim.SGD), "Optimizer has not the expected type"
    assert optimizer.defaults['lr'] == 0.001, "Learning rate is not correct"
    assert optimizer.defaults['weight_decay'] == 0.0001, "Weight decay is not correct"
    assert optimizer.defaults['momentum'] == 0.9, "Weight decay is not correct"

def test_prepare_training_objects(mock_train_parameters):

    targets = ["cat", "dog"]
    model, optimizer, device, loss_function = prepare_training_objects(targets,mock_train_parameters)
    assert model is not None, "The model has not been initialized correctly"
    assert optimizer is not None, "The optimizer has not been initialized correctly"
    assert device == torch.device("cuda" if torch.cuda.is_available() else "cpu"), "The device is not correct"
    assert loss_function is not None, "The loss function has not been initialized correctly"


def test_train(mock_train_parameters,best_model):
    """
    Function to test the training process
    """
    #Setup
    optimizer = torch.optim.Adam(best_model.parameters(), lr=mock_train_parameters["learning_rate"],
                                 weight_decay=mock_train_parameters["weight_decay"])
    device = torch.device("cpu")
    loss_function = torch.nn.CrossEntropyLoss()

    _, loss_per_epoch = training(mock_train_parameters, best_model, optimizer, device, loss_function)
    
    # Verificar que se entrena y que la pérdida por epoch está decreciendo
    assert len(loss_per_epoch) == mock_train_parameters["num_epochs"], "Epochs don't match"
    assert loss_per_epoch[0] > loss_per_epoch[-1], "The loss is not decreasing"


#EVALUATE SCRIPT TESTS

def test_load_image(sample_image):
    """
    Function to test the load_image function
    """
    img = load_image(sample_image)
    assert isinstance(img, Image.Image), "The function does not return an image"
    assert img.mode == "RGB", "The image is not in RGB mode"




def test_preapare_validation_dataloaders():
    """
    Function to test that the DataLoader is correctly generated
    """

    batch_size = 32
    valid_dataloader = preapare_validation_dataloaders(
                        PROCESSED_VALID_IMAGES,
                        batch_size
                    )
    
    batch = next(iter(valid_dataloader))
    print(batch[0].shape)
    assert len(batch) == 2, "DataLoader has not the expected number of elements"
    assert batch[0].shape[0] == batch_size, "The batch size is not correct"
    assert batch[0].shape[2] == 224, "The height of the image is not correct"
    assert batch[0].shape[3] == 224, "The width of the image is not correct"
    assert batch[0].shape[1] == 3, "The number of channels is not correct"
    assert len(valid_dataloader) > 0, "Dataloader does not have any data"




def test_evaluate_model(best_model):
    """
    It tests the evaluate_model function
    """
    batch_size = 32
    accuracy = evaluate_model(best_model, batch_size)
    assert isinstance(accuracy, float), "Accuracy should be a float"
    assert 0 <= accuracy <= 100, "Accuracy should be between a percentage"




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









    




