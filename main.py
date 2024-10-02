from pathlib import Path
import os
import sys

from src.features_data.prepare import main_preprocessing
from src.modeling.test import main_test
from src.modeling.evaluate import main_validation
from src.modeling.train import main_train
<<<<<<< HEAD
<<<<<<< HEAD
from src.config import PREPROCESSING,TRAINING,MODELS_DIR,PROJ_ROOT
=======
from src.config import PREPROCESSING,TRAINING,MODELS_DIR
>>>>>>> 63434e005eb1c3a27bcd639d601a33dc3d5366d1
=======
from src.config import PREPROCESSING,TRAINING,MODELS_DIR
>>>>>>> 63434e005eb1c3a27bcd639d601a33dc3d5366d1



def main():
<<<<<<< HEAD
<<<<<<< HEAD
    print(PROJ_ROOT)
=======
>>>>>>> 63434e005eb1c3a27bcd639d601a33dc3d5366d1
=======
>>>>>>> 63434e005eb1c3a27bcd639d601a33dc3d5366d1

    print("Starting data preprocessing...")
    if PREPROCESSING:
        main_preprocessing()

    if not TRAINING:
        #Check if there are models to test
        pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        if len(pkl_files) == 0:
            print("No models to evaluate")
            sys.exit()
    else: 
        print("Starting training...")
        parameters_dict, run_ids= main_train()
        print("Starting validation...")
        main_validation(parameters_dict,run_ids)
    
    print("Starting testing...")
    main_test()

if __name__ == "__main__":
    main()