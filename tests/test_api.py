import pytest
from fastapi.testclient import TestClient
from http import HTTPStatus
from unittest.mock import MagicMock, patch
from src.app.api import app, allowed_file_format, image_to_tensor, save_rating_api_to_csv, save_rating_models_to_csv
from pathlib import Path
from src.config import TEST_IMAGES_DIR
from fastapi import UploadFile
import io

# Utilizamos TestClient para simular peticiones a la API
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

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
        mock_file = MagicMock(filename=f"{TEST_IMAGES_DIR}/test_image.{img_format}")
        assert allowed_file_format(mock_file), f"El formato {img_format} debería ser permitido."
    
    # Prueba con un formato no permitido
    mock_file_invalid = MagicMock(filename=f"{TEST_IMAGES_DIR}/test_image.txt")
    assert not allowed_file_format(mock_file_invalid), "El formato txt no debería ser permitido."


def test_image_to_tensor():
    #Confirma que las dimensiones del tensor son correctas
    image_path = TEST_IMAGES_DIR / "test_image.jpeg"
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    tensor = image_to_tensor(img_bytes)
    assert tensor.shape == (1, 3, 224, 224), "El tensor no tiene las dimensiones esperadas"


def test_rate_api():
    # Simular la función para guardar la calificación en un archivo CSV
    with patch("src.app.api.save_rating_api_to_csv") as mock_save:
        mock_save.return_value = None

        response = client.post("/rate_api", json={"rating": 5})
        assert response.status_code == HTTPStatus.OK
        assert response.json()["message"] == "Rating added successfully"
        mock_save.assert_called_once_with(5)

def test_rate_model():
    # Simular la función para guardar la calificación del modelo en un archivo CSV
    with patch("src.app.api.save_rating_models_to_csv") as mock_save:
        mock_save.return_value = None

        response = client.post("/models/rate/Model_1", json={"rating": 4})
        assert response.status_code == HTTPStatus.OK
        assert response.json()["message"] == "Rating added successfully"
        mock_save.assert_called_once_with("Model_1", 4)

def test_get_models_list(client):
    # Simular la carga de modelos desde el archivo JSON
    mock_data = {"models": ["Model_1", "Model_2", "Model_3"]}
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '{"models": ["Model_1", "Model_2", "Model_3"]}'
        response = client.get("/models_available")
        assert response.status_code == HTTPStatus.OK
        assert "data" in response.json()
        assert response.json()["data"] == mock_data

def test_model_stats():
    # Datos de prueba
    model_name = "Model_1"
    model_summary_mock = "Model summary here"
    
    # Simular la lista global de modelos
    with patch("app.model_list", [MagicMock() for _ in range(3)]):
        # Simular la función get_model_summary para devolver datos simulados
        with patch("app.get_model_summary", return_value=model_summary_mock):
            # Hacer la solicitud al endpoint con el nombre del modelo correcto
            response = client.get(f"/model-summary?Model_name={model_name}")
            
            # Verificar el código de estado y la respuesta
            assert response.status_code == HTTPStatus.OK
            assert model_summary_mock in response.text
            assert "Status Code: 200" in response.text

def test_health_check_model_loaded():
    model_name = "Model_1"
    
    # Simula la lista de modelos con al menos un modelo cargado
    with patch("app.model_list", [MagicMock() for _ in range(3)]):
        response = client.get(f"/health?Model_name={model_name}")
        
        assert response.status_code == HTTPStatus.OK
        json_response = response.json()
        assert json_response["status"] == "API is up and running!"
        assert json_response["model_status"] == "loaded"

# Para archivo de imagen individual
def test_predict_image_single_file():
    model_name = "Model_1"
    file_content = io.BytesIO(b"fake image data")
    file = UploadFile(filename="test_image.jpg", file=file_content)

    # Simula el formato de archivo permitido
    with patch("app.allowed_file_format", return_value=True):  
        response = client.post(f"/predict/{model_name}", files={"files": (file.filename, file_content, "image/jpeg")})

        assert response.status_code == HTTPStatus.OK
        assert "results" in response.json()
        assert response.json()["results"][0]["class"] in ["cat", "dog", "unknown"]  

# Para archivo zip
def test_predict_image_zip():
    model_name = "Model_1"
    # Simula un archivo zip
    zip_content = io.BytesIO()  

    with patch("app.zipfile.ZipFile") as mock_zip:
        # Simulamos el comportamiento de zipfile
        mock_zip.return_value.__enter__.return_value.infolist.return_value = [MagicMock(filename="test_image.jpg")]
        mock_zip.return_value.__enter__.return_value.read.return_value = b"fake image data"
        mock_zip.return_value.__enter__.return_value.read.side_effect = [b"fake image data"]  

        response = client.post(f"/predict/{model_name}", files={"files": ("test.zip", zip_content, "application/zip")})

        assert response.status_code == HTTPStatus.OK
        assert "results" in response.json()
        assert response.json()["results"][0]["class"] in ["cat", "dog", "unknown"]  

