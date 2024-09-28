from pathlib import Path
import os

from src.features_data.prepare import main_preprocessing
from src.modeling.train import main_train
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
def main():
    #Preprocessing
    if not any(file.endswith('.csv') for file in os.listdir(PROCESSED_DATA_DIR)):
        print("Starting data preprocessing...")
        main_preprocessing()

    #Trainig and validation
    print("Starting training and validation...")
    #main_train()


    #Test
    print("Starting testing...")

if __name__ == "__main__":
    main()