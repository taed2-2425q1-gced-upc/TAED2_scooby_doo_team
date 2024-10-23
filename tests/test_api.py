import pytest
from fastapi.testclient import TestClient
from http import HTTPStatus
from unittest.mock import MagicMock, patch
from src.app.api import app, allowed_file_format, image_to_tensor, save_rating_api_to_csv, save_rating_models_to_csv
from pathlib import Path
from src.config import TEST_IMAGES_DIR

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
    with patch("builtins.open", new_callable=MagicMock):
        response = client.get("/models_available")
        assert response.status_code == HTTPStatus.OK
        assert "data" in response.json()
