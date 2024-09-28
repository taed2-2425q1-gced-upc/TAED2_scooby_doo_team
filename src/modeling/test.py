from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import pickle

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_data(features_path: Path) -> pd.DataFrame:
    """
    Handles the loading of test features from a CSV file and provides
    detailed error handling.
    """
    logger.info(f"Loading test features from {features_path}...")
    try:
        data = pd.read_csv(features_path)
        logger.info(f"Successfully loaded {len(data)} samples from {features_path}.")
        return data
    except FileNotFoundError:
        logger.error(f"File {features_path} not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File {features_path} is empty.")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def load_model(model_path: Path):
    """
    Loads the machine learning model from a pickle file and includes handling for 
    file errors or unpickling issues.
    """
    logger.info(f"Loading model from {model_path}...")
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logger.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error(f"Model file {model_path} not found.")
        raise
    except pickle.UnpicklingError:
        logger.error(f"Error unpickling the model at {model_path}.")
        raise
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        raise

def save_predictions(predictions, predictions_path: Path):
    """
    Saves the predictions to a CSV file and ensures errors during the saving 
    process are properly caught.

    """
    logger.info(f"Saving predictions to {predictions_path}...")
    try:
        pd.DataFrame(predictions, columns=["Prediction"]).to_csv(predictions_path, index=False)
        logger.success(f"Predictions saved successfully to {predictions_path}.")
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    """
    1. Loads test features from a CSV file.
    2. Loads the pre-trained model (pickled).
    3. Performs inference to generate predictions.
    4. Saves the predictions to a CSV file.
    """
    logger.info("Starting model evaluation pipeline...")

    # ----  Load test data ----
    try:
        test_features = load_data(features_path)
    except Exception:
        logger.error("Failed to load test features. Exiting.")
        return
    
    # ---- Load model ----

    try:
        model = load_model(model_path)
    except Exception:
        logger.error("Failed to load the model. Exiting.")
        return
    
    # ---- Perform inference ----
    logger.info("Performing inference for model...")
    try:
        # Check if model has a batch prediction capability
        if hasattr(model, "predict"):
            # If the model supports batch prediction, perform it on the entire dataset
            predictions = []
            for i in tqdm(range(len(test_features)), total=len(test_features)):
                if pd.isnull(test_features).any():
                    logger.warning(f"Sample {i} contains missing values: {test_features}")

                # Assuming the model has a 'predict' method
                try:
                    prediction = model.predict([test_features.iloc[i]])  # Excluding the index if necessary
                    predictions.append(prediction[0])
                    
                except ValueError as ve:
                    logger.error(f"ValueError occurred during prediction for sample {i}: {ve}")    
                except Exception as e:
                    logger.error(f"Prediction failed for sample {i}: {e}")
                
            logger.success("Inference completed successfully.")
        else:
            logger.error("The loaded model does not have a 'predict' method.")
            return
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return
    # ---- Save predictions ----
    try:
        save_predictions(predictions, predictions_path)
    except Exception:
        logger.error("Failed to save predictions. Exiting.")
        return

    logger.info("Model evaluation pipeline completed.")

if __name__ == "__main__":
    app()
