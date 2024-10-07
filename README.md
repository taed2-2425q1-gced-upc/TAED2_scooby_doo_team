# TAED2_scooby_doo_team

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Software engineering for ML systems

## Project Organization


```

├── README.md                         <- The top level README
├── pyproject.toml                    <- Defines dependencies of python libraries
├── poetry.lock                       <- Defines which versions of
├── params.yaml                       <- Defines which models to use and which combinations of hyperparameters to train
├── .*                                <- Other configuration files (such as .gitignore, .dvcignore, etc), not stated for simplicity
├── data                              <- Folder containing the data to train and validate the model
│   ├── processed                     <- The final data processed for model use
│   │   ├── train_images              <- Images used during training
│   │   │   ├── cats                  <- Images of cats
│   │   │   └── dogs                  <- Images of dogs
│   │   ├── valid_images              <- Images used during validation
│   │   │   ├── cats                  <- Images of cats
│   │   │   └── dogs                  <- Images of dogs
│   │   └── *.csv                     <- Files generated during processing of data such as target class of each image
│   └── raw                           <- The original, immutable data dump, stored in parquet files
├── docs                              <- Documents such as model card or data card
├── metrics                           <- Dumps of metrics such as the accuracy of the best model
├── reports                           <- Generated reports such as the ones generated through deepchecks
├── tests                             <- Tests to be used with pytest
│   ├── test_models.py                <- Tests for the generated models
│   └── test_data                     <- Data used during the tests
│       ├── cats                      <- Images of cats
│       └── dogs                      <- Images of dogs
├── models                            <- Compressed dumps of the trained models
└── src                               <- Source code for use in this project
    ├── __init__.py                   <- Makes TAED2_scooby_doo_team a Python module
    ├── config.py                     <- Configuration file for python
    ├── features_data                 <- Files for feature extraction and data processing
    │   ├── deepchecks_validation.py  <- Performs data checks and writes a report in reports
    |   └── prepare.py                <- Creates the dataset for trainig
    └── models                        <- Files for model training and evaluation
        ├── evaluate.py               <- Validation of the model
        ├── test.py                   <- Test the model
        └── train.py                  <- Train the model

```


--------

