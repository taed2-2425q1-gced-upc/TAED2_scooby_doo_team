from pathlib import Path
import pandas as pd
import yaml
import os
import io
from PIL import Image
from sklearn.model_selection import train_test_split

# Importar las rutas desde src.config
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, PROCESSED_TRAIN_IMAGES, PROCESSED_VALID_IMAGES, PROCESSED_TEST_IMAGES



def load_data(input_folder_path):
    """Carga los archivos Parquet y concatena los DataFrames."""
    data_file_1 = input_folder_path / "train-00000-of-00002.parquet"
    data_file_2 = input_folder_path / "train-00001-of-00002.parquet"

    df1 = pd.read_parquet(data_file_1)
    df2 = pd.read_parquet(data_file_2)

    return pd.concat([df1, df2], ignore_index=True)



def load_params_prepare(params_path):
    """Carga los parámetros desde un archivo YAML."""
    with open(params_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
            return params["prepare"]
        except yaml.YAMLError as exc:
            print(exc)
            return None



def split_data(df, params):
    """Separa los datos en conjuntos de entrenamiento, validación y prueba."""
    y = df["labels"]
    X = df.drop(["labels"], axis=1)

    # Dividir los datos en entrenamiento y un conjunto temporal
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=params["train_size"],
        test_size=params["test_size"],  
        random_state=params["random_state"],
    )

    # Dividir el conjunto temporal en validación y prueba
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - params["valid_size"],   # Ajustar según el tamaño total
        random_state=params["random_state"],
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test



def save_images(dataframe, prepared_images_path, prefix):
    """
    Saves images from a dataframe to a folder
    """
    Path(prepared_images_path).mkdir(exist_ok=True)
    image_index = 0
    for _, row in dataframe.iterrows():
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        img.save(prepared_images_path / f"{prefix}_{image_index}.jpg")
        image_index += 1



def save_data(X_train, y_train, X_valid, y_valid, X_test, y_test, prepared_folder_path,prepared_train_images_path, prepared_valid_images_path, prepared_test_images_path):
    """Guarda los conjuntos de datos en archivos CSV."""
    Path(prepared_folder_path).mkdir(exist_ok=True)
    print('hola')

    X_train.to_csv(prepared_folder_path / "X_train.csv", index=False)
    y_train.to_csv(prepared_folder_path / "y_train.csv", index=False)
    X_valid.to_csv(prepared_folder_path / "X_valid.csv", index=False)
    y_valid.to_csv(prepared_folder_path / "y_valid.csv", index=False)
    X_test.to_csv(prepared_folder_path / "X_test.csv", index=False)
    y_test.to_csv(prepared_folder_path / "y_test.csv", index=False)


    save_images(X_train, prepared_train_images_path, "image_train")
    save_images(X_valid, prepared_valid_images_path, "image_valid")
    save_images(X_test, prepared_test_images_path, "image_test")

    print("Data saved successfully.")




"""Función principal."""
    
params_path = Path("params.yaml")
input_folder_path = RAW_DATA_DIR
prepared_folder_path = PROCESSED_DATA_DIR
prepared_train_images_path = PROCESSED_TRAIN_IMAGES
prepared_valid_images_path = PROCESSED_VALID_IMAGES
prepared_test_images_path = PROCESSED_TEST_IMAGES

df = load_data(input_folder_path)
params = load_params_prepare(params_path)

#To ensure use parameters
if params:
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df, params)
    save_data(X_train, y_train, X_valid, y_valid, X_test, y_test, prepared_folder_path, prepared_train_images_path, prepared_valid_images_path, prepared_test_images_path)
