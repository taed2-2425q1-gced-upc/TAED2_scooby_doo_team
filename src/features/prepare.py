from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Importar las rutas desde src.config
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Ruta del archivo de parámetros
params_path = Path("params.yaml")

# Ruta de la carpeta de datos de entrada
input_folder_path = RAW_DATA_DIR

# Rutas de los archivos parquet
data_file_1 = input_folder_path / "train-00000-of-00002.parquet"
data_file_2 = input_folder_path / "train-00001-of-00002.parquet"

# Cargar los dos archivos .parquet localmente
df1 = pd.read_parquet(data_file_1)
df2 = pd.read_parquet(data_file_2)

# Concatenar ambos DataFrames
df = pd.concat([df1, df2], ignore_index=True)

# Leer los parámetros de preparación de datos desde params.yaml
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["prepare"]
    except yaml.YAMLError as exc:
        print(exc)

# Separar la etiqueta (label) de los predictores
y = df["labels"]
X = df.drop(["labels"], axis=1)

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=params["train_size"],
    test_size=params["test_size"],
    random_state=params["random_state"],
)

# Crear el directorio de salida si no existe
prepared_folder_path = PROCESSED_DATA_DIR
Path(prepared_folder_path).mkdir(exist_ok=True)

# Guardar los conjuntos de entrenamiento y validación en archivos CSV
X_train_path = prepared_folder_path / "X_train.csv"
y_train_path = prepared_folder_path / "y_train.csv"
X_valid_path = prepared_folder_path / "X_valid.csv"
y_valid_path = prepared_folder_path / "y_valid.csv"

X_train.to_csv(X_train_path, index=False)
print(f"Writing file {X_train_path} to disk.")

y_train.to_csv(y_train_path, index=False)
print(f"Writing file {y_train_path} to disk.")

X_valid.to_csv(X_valid_path, index=False)
print(f"Writing file {X_valid_path} to disk.")

y_valid.to_csv(y_valid_path, index=False)
print(f"Writing file {y_valid_path} to disk.")
