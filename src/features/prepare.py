"""
This script contains the preprocessing functions to prepare the data for training.
"""

from pathlib import Path
import io
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (PROCESSED_DATA_DIR,
                        RAW_DATA_DIR,
                        PROCESSED_TRAIN_IMAGES,
                        PROCESSED_VALID_IMAGES,
                        PROCESSED_TEST_IMAGES
)



def load_data(input_folder_path):
    """
    Load the data from the input folder
    """
    data_file_1 = input_folder_path / "train-00000-of-00002.parquet"
    data_file_2 = input_folder_path / "train-00001-of-00002.parquet"

    df1 = pd.read_parquet(data_file_1)
    df2 = pd.read_parquet(data_file_2)

    return pd.concat([df1, df2], ignore_index=True)



def load_params_prepare(path):
    """
    Load the parameters from the params.yaml file
    """
    with open(path, "r", encoding="utf-8") as params_file:
        try:
            parameters = yaml.safe_load(params_file)
            return parameters["prepare"]
        except yaml.YAMLError as exc:
            print(exc)
            return None



def split_data(dataframe, parameters):
    """
    Separa los datos en conjuntos de entrenamiento, validación y prueba
    """
    y = dataframe["labels"]
    x = dataframe.drop(["labels"], axis=1)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        train_size=parameters["train_size"],
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
    )


    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=1 - parameters["valid_size"],
        random_state=parameters["random_state"],
    )
    return x_train, x_valid, x_test, y_train, y_valid, y_test



def save_images(dataframe,labels, prepared_images_path, prefix):
    """
    Saves images from a dataframe to a folder
    """
    Path(PROCESSED_DATA_DIR/"dogs").mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_DATA_DIR/"cats").mkdir(parents=True, exist_ok=True)

    num_cats = 0
    num_dogs = 0
    dataframe_index = 0
    for _, row in dataframe.iterrows():
        label = labels.iloc[dataframe_index]
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        if label == 0:
            img.save(prepared_images_path /"cats"/f"{prefix}_{num_cats}.jpg")
            num_cats += 1
        else:
            img.save(prepared_images_path /"dogs"/f"{prefix}_{num_dogs}.jpg")
            num_dogs += 1
        dataframe_index += 1



def save_data(x_train, y_train, x_valid, y_valid, x_test, y_test):
    """Guarda los conjuntos de datos en archivos CSV."""
    Path(PROCESSED_DATA_DIR).mkdir(exist_ok=True)

    x_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    x_valid.to_csv(PROCESSED_DATA_DIR / "X_valid.csv", index=False)
    y_valid.to_csv(PROCESSED_DATA_DIR / "y_valid.csv", index=False)
    x_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)


    save_images(x_train,y_train,PROCESSED_TRAIN_IMAGES, "image_train")
    save_images(x_valid,y_valid,PROCESSED_VALID_IMAGES, "image_valid")
    save_images(x_test,y_test ,PROCESSED_TEST_IMAGES, "image_test")

    print("Data saved successfully.")



"""Función principal"""

params_path = Path("params.yaml")

df = load_data(RAW_DATA_DIR)
params = load_params_prepare(params_path)

if params:
    x_train_data, x_valid_data, x_test_data, y_train_data, y_valid_data, y_test_data = split_data(df, params)
    save_data(x_train_data, y_train_data, x_valid_data,
              y_valid_data, x_test_data, y_test_data
              )
