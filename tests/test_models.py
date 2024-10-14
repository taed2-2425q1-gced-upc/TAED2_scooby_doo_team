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

from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_TRAIN_IMAGES,PROCESSED_TEST_IMAGES,PROCESSED_VALID_IMAGES,MODELS_DIR
from src.models.train import load_params_train
from src.models.train import combine_hyperparameters
from src.models.train import get_model, get_optimizer
from src.models.train import preapare_train_dataloaders
from src.models.train import training
from src.models.evaluate import preapare_validation_dataloaders
from src.models.evaluate import load_image
from src.models.evaluate import evaluate_model


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




@pytest.fixture
def mock_yaml_file(tmp_path):
    """
    Returns the path to a mock yaml file
    """

    yaml_content = """
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


@pytest.fixture
def sample_image(tmp_path):
    """
    Returns the path to a sample image
    """
    # Crear una imagen temporal para usar en el test
    img_path = "data/processed/valid_images/dogs/image_valid_0.jpg"
    return img_path



#TRAINING SCRIPT TESTS

def test_load_params_train(mock_yaml_file):
    """
    Function to check if the params are correctly loaded from the load_params_train function
    """

    params = load_params_train(mock_yaml_file)
    assert params is not None, "Los parámetros no se cargaron correctamente."
    assert params["algorithm"] == "VisualTransformer", "Algorithm is not correct."
    assert params["model_name"] == "google/vit-base-patch16-224", "Model name is not correct."
    assert params["batch_size"] == 32, "Batch size is not correct."





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




def test_get_model():
    """
    Function to test that in get_model funtion, the model is correctly initialized
    """

    model = get_model("VisualTransformer", "google/vit-base-patch16-224", ["cat", "dog"])
    assert model is not None, "The model has not been initialized correctly"
    assert model.config.num_labels == 2, "Number of labels is not correct."



def test_get_optimizer():
    """
    Function to test that in get_optimizer funtion, the optimizer is correctly initialized
    """

    model = get_model("VisualTransformer", "google/vit-base-patch16-224", ["cat", "dog"])
    #Example of parameters for th"""e optimizer if the optimizer is adam
    optimizer = get_optimizer("adam", 0.001, 0.0001, 0.9, model)
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer has not the expected type"
    assert optimizer.defaults['lr'] == 0.001, "Learning rate is not correct"
    assert optimizer.defaults['weight_decay'] == 0.0001, "Weight decay is not correct"
    assert 'momentum' not in optimizer.defaults, "Momentum is not correct"
    #Example of parameters for the optimizer if the optimizer is sgd
    optimizer = get_optimizer("sgd", 0.001, 0.0001, 0.9, model)
    assert isinstance(optimizer, torch.optim.SGD), "Optimizer has not the expected type"
    assert optimizer.defaults['lr'] == 0.001, "Learning rate is not correct"
    assert optimizer.defaults['weight_decay'] == 0.0001, "Weight decay is not correct"
    assert optimizer.defaults['momentum'] == 0.9, "Weight decay is not correct"




def test_train(mock_train_parameters):
    """
    Function to test the training process
    """
    #Setup
    model = get_model(mock_train_parameters["algorithm"], mock_train_parameters["model_name"], ["cat", "dog"])
    optimizer = get_optimizer(
        mock_train_parameters["optimizer"],
        mock_train_parameters["learning_rate"],
        mock_train_parameters["weight_decay"],
        mock_train_parameters["momentum"],
        model
    )
    device = torch.device("cpu")  # para el test usaremos CPU
    loss_function = torch.nn.CrossEntropyLoss()

    model, loss_per_epoch = training(mock_train_parameters, model, optimizer, device, loss_function)
    
    # Verificar que se entrena y que la pérdida por epoch está decreciendo
    assert len(loss_per_epoch) == mock_train_parameters["num_epochs"], "Epochs don't match"
    assert loss_per_epoch[0] > loss_per_epoch[-1], "La pérdida no está decreciendo después de entrenar."



#EVALUATE SCRIPT TESTS

def test_load_image(sample_image):
    """
    Function to test the load_image function
    """
    img = load_image(sample_image)
    assert isinstance(img, Image.Image), "La función no retornó una instancia de imagen."
    assert img.mode == "RGB", "La imagen no está en modo RGB."



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



def test_evaluate_model():
    """
    It tests the evaluate_model function
    """
    run_name = "MODEL_1.pkl"
    with open(MODELS_DIR / run_name, "rb") as pickled_model:
            model = pickle.load(pickled_model)
    batch_size = 32
    accuracy = evaluate_model(model, batch_size)
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









    




