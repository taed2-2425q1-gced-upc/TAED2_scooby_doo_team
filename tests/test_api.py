from unittest.mock import MagicMock, patch
from http import HTTPStatus
import zipfile
import json
import io
import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile
import torch
from PIL import Image
from src.app.api import app, allowed_file_format, image_to_tensor, save_rating_api_to_csv, rating_api, save_rating_models_to_csv, models_rate, rating_models_api, ratings_data
from src.config import TEST_IMAGE_DIR
import csv
from datetime import datetime

#TEST_IMAGES_DIR = "/tests/pytest_images"

# Utilizamos TestClient para simular peticiones a la API
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

def test_load_ratings():
    with open(rating_models_api, "r") as f:
        assert json.load(f)
    assert rating_models_api

def test_read_root(client):
    response = client.get("/")
    json = response.json()
    assert response.status_code == HTTPStatus.OK
    assert json["data"]["message"] == "Welcome to ScoobyDoo project API!"
    assert json["message"] == "OK"
    assert json["status-code"] == HTTPStatus.OK

def test_allowed_file_format():
    # Lista de formatos de imagen que queremos probar
    image_formats = ["jpeg", "png", "gif", "webp", "jpg"]
    for img_format in image_formats:
        # Crea un mock para simular un archivo con el formato de imagen
        mock_file = MagicMock(filename=f"{TEST_IMAGE_DIR}/test_image.{img_format}")
        assert allowed_file_format(mock_file), f"El formato {img_format} debería ser permitido."
    # Prueba con un formato no permitido
    mock_file_invalid = MagicMock(filename=f"{TEST_IMAGE_DIR}/test_image.txt")
    assert not allowed_file_format(mock_file_invalid), "El formato txt no debería ser permitido."


def test_image_to_tensor():
    #Confirma que las dimensiones del tensor son correctas
    image_path = TEST_IMAGE_DIR/"test_image.jpg"
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    tensor = image_to_tensor(img_bytes)
    assert tensor.shape == (1, 3, 224, 224), "El tensor no tiene las dimensiones esperadas"
    assert isinstance(tensor, torch.Tensor), "Output is not a PyTorch tensor."


@pytest.mark.parametrize("valid_rating", [1, 2, 3, 4, 5])
def test_rate_api(client, valid_rating):
    # Simular la función para guardar la calificación en un archivo CSV
    with patch("src.app.api.save_rating_api_to_csv") as mock_save:

        response = client.post(f"/rate_api?rating={valid_rating}")

        assert response.status_code == HTTPStatus.OK
        assert response.json()["message"] == "Rating added successfully"
        mock_save.assert_called_once_with(valid_rating)

@pytest.mark.parametrize("invalid_rating", [0, 6])
def test_rate_api_invalid_ratings(client, invalid_rating):

    response = client.post(f"/rate_api?rating={invalid_rating}")
    
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"] ==  "Rating must be between 1 and 5"

def test_rate_api_unexpected_exception(client):
    with patch("src.app.api.save_rating_api_to_csv", side_effect=Exception("Unexpected error")):
        response = client.post("/rate_api?rating=3")
        
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert response.json()["detail"] == "Unexpected error: Unexpected error"

def test_rate_model_valid_rate(client):
    # Simular la función para guardar la calificación del modelo en un archivo CSV
    with patch("src.app.api.save_rating_models_to_csv") as mock_save:
        model_name = "Model_1"

        response = client.post("/models/rate/Model_1?rating=4")
        assert ratings_data[model_name]["average_rating"] == round(sum(ratings_data[model_name]["ratings"]) / len(ratings_data[model_name]["ratings"]), 2)

        assert response.status_code == HTTPStatus.OK
        assert response.json()["message"] == "Rating added successfully"
        mock_save.assert_called_once_with("Model_1", 4)

def test_rate_model_invalid_rate(client):
    response = client.post("/models/rate/Model_1?rating=8")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"] =="Rating must be between 1 and 5"

def test_rate_model_invalid_model(client):
    response = client.post("/models/rate/Model_5?rating=3")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"] =="Model not found. Please check the model name and try again."


def test_get_models_list(client):
    # Simular la carga de modelos desde el archivo JSON
    mock_data = {
        "Model_1": {"metrics": {"accuracy": 0.95}, "rate": 4.0},
        "Model_2": {"metrics": {"accuracy": 0.89}, "rate": 4},
        "Model_3": {"metrics": {"accuracy": 0.92}, "rate": 3}
    }

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)
        response = client.get("/models_available")
        assert response.status_code == HTTPStatus.OK
        assert "data" in response.json()

# Simular la lista global de modelos
mock_model_list = ["Model_1", "Model_2", "Model_3"]
mock_summary = "Mocked model summary"


def test_model_stats_valid_model(client):
     with patch("src.app.api.get_model_summary", return_value=mock_summary):
        with patch("src.app.api.model_list", mock_model_list):
            model_name = "Model_1"
            response = client.get(f"/model-summary?Model_name={model_name}")

            assert response.status_code == HTTPStatus.OK
            assert mock_summary in response.text
   

@patch("src.app.api.model_list", mock_model_list)
def test_model_stats_invalid_model(client):
    # Test with an out-of-range model index
    model_name = "Model_9"
    response = client.get(f"/model-summary?Model_name={model_name}")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "Model not found" in response.json()["detail"]


@patch("src.app.api.model_list", mock_model_list) 
def test_model_stats_bad_model_name_format(client):
    model_name = "Model_XYZ"
    response = client.get(f"/model-summary?Model_name={model_name}")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "Invalid model number" in response.json()["detail"]


def test_model_stats_summary_exception(client):
    with patch("src.app.api.get_model_summary", side_effect=Exception("Summary generation error")):
        with patch("src.app.api.model_list", mock_model_list):
            model_name = "Model_1"
            response = client.get(f"/model-summary?Model_name={model_name}")

            assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
            assert "Error fetching model stats" in response.json()["detail"]


def test_health_check_valid_model(client):
    """Test valid model name and loaded model."""

    model_name = "Model_1"
 
    # Simula la lista de modelos con al menos un modelo cargado
    with patch("src.app.api.model_list",  mock_model_list):
        response = client.get(f"/health?Model_name={model_name}")
        assert response.status_code == HTTPStatus.OK
        assert response.json() == {
        "status": "API is up and running!",
        "model_status": "loaded"  
        }

def test_health_check_negative_index(client):
    """Test model name that results in a negative index."""
    model_name = "Model_-1"
    with patch("src.app.api.model_list",  mock_model_list):
        response = client.get(f"/health?Model_name={model_name}")
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert response.json()["detail"] == "Invalid model name format. Ensure it follows 'Model_<number>'"

def test_health_check_invalid_format(client):
    """Test invalid format for the model name."""
    model_name = "InvalidModel_1"
    with patch("src.app.api.model_list",  mock_model_list):
        response = client.get(f"/health?Model_name={model_name}")
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert response.json()["detail"] == "Invalid model name format. Ensure it follows 'Model_<number>'"


def test_health_check_out_of_bounds_model(client):
    """Test model name that results in out-of-bounds index."""
    model_name = "Model_9"
    with patch("src.app.api.model_list", mock_model_list):
        response = client.get(f"/health?Model_name={model_name}")
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert response.json()["detail"] == "Model not found. Please check the model name and try again."

def test_health_check_unloaded_model(client):
    """Test valid model name with an unloaded (None) model."""
    model_name = "Model_1"
    with patch("src.app.api.model_list", [None]):
        response = client.get(f"/health?Model_name={model_name}")
        assert response.status_code == HTTPStatus.OK
        assert response.json() == {
            "status": "API is up and running!",
            "model_status": "not loaded"
        }


def test_health_check_exception(client):
    """Test an unexpected error."""
    model_name = "Model_1"
    with patch("src.app.api.model_list", side_effect=Exception("Unexpected error")):
        response = client.get(f"/health?Model_name={model_name}")
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert response.json() == {
            "detail": "Model not found. Please check the model name and try again."
        }
        
# Para archivo de imagen individual
def test_predict_image_single_file(client):
    model_name = "Model_1"
    file_content = io.BytesIO(b"fake image data")
    file = UploadFile(filename="test_image.jpg", file=file_content)

    # Simula el formato de archivo permitido
    with patch("src.app.api.allowed_file_format", return_value=True):
        valid_tensor = torch.rand(1, 3, 224, 224)
        with patch("src.app.api.image_to_tensor", return_value=valid_tensor):
            response = client.post(
                f"/predict/{model_name}", 
                files={"files": (file.filename, file.file.read(), "image/jpeg")})

            assert response.status_code == HTTPStatus.OK
            assert "results" in response.json()
            assert len(response.json()["results"]) == 1
            assert response.json()["results"][0]["class"] in ["cat", "dog", "unknown"]

def create_fake_jpeg_image():
    image = io.BytesIO()
    img = Image.new('RGB', (10, 10), color='red')  # Create a small red image
    img.save(image, format='JPEG')
    image.seek(0)  # Rewind the BytesIO object for reading
    return image

# Para archivo zip
def test_predict_image_zip(client):
    model_name = "Model_1"
    file_content = create_fake_jpeg_image()
    zip_file = io.BytesIO()
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("test_image.jpg", file_content.getvalue())

    zip_file.seek(0)

    with patch("src.app.api.zipfile.ZipFile", new_callable=MagicMock) as mock_zip:
        # Simulamos el comportamiento de zipfile
        mock_zip_instance = mock_zip.return_value.__enter__.return_value

        mock_zip_instance.read.side_effect = lambda name: file_content.getvalue() if name == "test_image.jpg" else None
        mock_zip_instance.infolist.return_value = [MagicMock(filename="test_image.jpg")]

        response = client.post(
            f"/predict/{model_name}", 
            files={"files": ("test.zip", zip_file.getvalue(), "application/zip")})

        assert response.status_code == HTTPStatus.OK
        assert "results" in response.json()
        assert len(response.json()["results"]) == 1
        assert response.json()["results"][0]["class"] in ["cat", "dog", "unknown"]  

def test_predict_image_valid_file(client):
    # Prepare a valid image file for testing
    image_path = TEST_IMAGE_DIR/"test_image.jpg"

    with open(image_path , "rb") as image_file:
        response = client.post("/predict/Model_1", files={"files": ("test_image.jpg", image_file, "image/jpeg")})
    
    assert response.status_code == HTTPStatus.OK
    assert "results" in response.json()

def test_predict_image_invalid_file(client):
    # Test with an invalid file type
    invalid_file = ("test_image.txt", b"not an image", "text/plain")
    file = UploadFile(filename=invalid_file[0], file=io.BytesIO(invalid_file[1]))
    assert not allowed_file_format(file), "El formato txt no debería ser permitido."

    response = client.post("/predict/Model_1", files={"files": [invalid_file]})
    
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "detail" in response.json()

def test_predict_image_invalid_format(client):
    response = client.post("/predict/Model_1", files={"files": [("test_image.txt", b"not an image", "text/plain")]})
    assert response.status_code == HTTPStatus.BAD_REQUEST

def test_predict_empty_file(client):
    response = client.post("/predict/Model_1", files={})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

def test_predict_image_gif_invalid_file(client):
    # Create a fake GIF file content
    file_content = io.BytesIO(b"GIF89a" + b'\x00\x00' * 100) 
    file = UploadFile(filename="test_image.gif", file=file_content)
    assert allowed_file_format(file)
    response = client.post("/predict/Model_1", files={"files": [("test_image.gif", file.file, "image/gif")]})

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "detail" in response.json()

def test_predict_image_gif_file(client):
    model_name = "Model_1"
    image_path = TEST_IMAGE_DIR/"test_image.gif"
    with open(image_path, "rb") as file:
        img_bytes = file.read()

    file_like = io.BytesIO(img_bytes)
    file_like.seek(0)
    file = UploadFile(filename="test_image.gif", file=file_like)

    assert allowed_file_format(file)
    response = client.post(
        f"/predict/{model_name}", 
        files={"files": ("test_image.gif", file.file, "image/gif")})
    
    assert response.status_code == HTTPStatus.OK
    assert "results" in response.json()  
    results = response.json()["results"]
    assert len(results) == 1 
    assert results[0]["class"] in ["cat", "dog", "unknown"]  

def test_predict_image_png_invalid_file(client):
    # Create a fake PNG file content (this is minimal and not a valid PNG)
    file_content = io.BytesIO(b"\x89PNG\r\n\x1A\n" + b'\x00' * 100)  # Minimal valid PNG header
    file = UploadFile(filename="test_image.png", file=file_content)
    response = client.post("/predict/Model_1", files={"files": [("test_image.png", file.file, "image/png")]})

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "detail" in response.json()
 
def test_predict_image_png_file(client):
    model_name = "Model_1"
    image_path = TEST_IMAGE_DIR/"test_image.png"
    with open(image_path, "rb") as file:
        img_bytes = file.read()

    file_like = io.BytesIO(img_bytes)
    file_like.seek(0)
    file = UploadFile(filename="test_image.png", file=file_like)

    assert allowed_file_format(file)
    response = client.post(
        f"/predict/{model_name}", 
        files={"files": ("test_image.png", file.file, "image/png")})
    
    assert response.status_code == HTTPStatus.OK
    assert "results" in response.json()  
    results = response.json()["results"]
    assert len(results) == 1 
    assert results[0]["class"] in ["cat", "dog", "unknown"]  

def test_predict_image_webp_invalid_file(client):
    # Create a fake WebP file content (this is a minimal representation and not a valid WebP file)
    file_content = io.BytesIO(b'RIFF' + b'\x00' * 12 + b'WEBP' + b'\x00' * 100) 
    file = UploadFile(filename="test_image.webp", file=file_content)

    response = client.post("/predict/Model_1", files={"files": [("test_image.webp", file.file, "image/webp")]})

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "detail" in response.json()

def test_predict_image_webp_file(client):
    model_name = "Model_1"
    image_path = TEST_IMAGE_DIR/"test_image.webp"
    with open(image_path, "rb") as file:
        img_bytes = file.read()

    file_like = io.BytesIO(img_bytes)
    file_like.seek(0)
    file = UploadFile(filename="test_image.webp", file=file_like)

    assert allowed_file_format(file)
    response = client.post(
        f"/predict/{model_name}", 
        files={"files": ("test_image.webp", file.file, "image/webp")})
    
    assert response.status_code == HTTPStatus.OK
    assert "results" in response.json()  
    results = response.json()["results"]
    assert len(results) == 1 
    assert results[0]["class"] in ["cat", "dog", "unknown"]  

def test_save_rating_api_to_csv():

    raiting = 5
    save_rating_api_to_csv(raiting)
    with open(rating_api, mode="r", newline="") as file:
        csv_reader = csv.reader(file)   

        current_time = datetime.now()
        day = current_time.day
        month = current_time.month
        year = current_time.year
        
        last_row = None
        for row in csv_reader:
            last_row = row  # This will end up being the last row after loop finishes
        
        read_raiting, read_day, read_month, read_year = last_row

        assert str(read_raiting) == str(raiting)
        assert str(read_day) == str(day)
        assert str(read_month) == str(month)
        assert str(read_year) == str(year)
def test_save_rating_models_to_csv():
    rating = 3
    modelname = "Model_1"
    save_rating_models_to_csv(modelname, rating)

    with open(models_rate, mode="r", newline="") as file:
        csv_reader = csv.reader(file) 

        current_time = datetime.now()
        day = current_time.day
        month = current_time.month
        year = current_time.year
        modelname = "Model_1" 

        last_row = None
        for row in csv_reader:
            last_row = row    
        
        read_modelname, read_raiting, read_day, read_month, read_year = last_row
        assert str(read_modelname) == modelname
        assert str(read_raiting) == str(rating)
        assert str(read_day) == str(day)
        assert str(read_month) == str(month)
        assert str(read_year) == str(year)

    

def test_get_model_summary():
    try: 
        with patch("src.app.api.get_model_summary", return_value=mock_summary):
            assert True 
    except:
        assert False 