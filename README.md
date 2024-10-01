# TAED2_scooby_doo_team

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Software engineering for ML systems

## Project Organization


```

├── README.md             <- The top level README
├── .*                    <- other configuration files & tracking files
├── data
│   ├── processed         <- The final, canonical data sets for modeling.
│   └── raw               <- The original, immutable data dump.
├── docs                  <- Documents such as model card and others
├── metrics               <- Metrics such as emissions and accuracy of the best model
├── models                <- Mlflow models (will also be saved at the servers)
└── src                   <- Source code for use in this project.
    │
    ├── __init__.py       <- Makes TAED2_scooby_doo_team a Python module
    ├── config.py         <- Configuration file for python
    ├── features_data     <- Files for feature extraction and data processing
    |   └── prepare.py    <- Creates the dataset for trainig
    └── modeling          <- Files for model training and evaluation
        ├── evaluate.py   <- Validation of the model
        ├── test.py       <- Train the model
        └── train.py      <- Test the model

```


--------

