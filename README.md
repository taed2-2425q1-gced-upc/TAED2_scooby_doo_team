# TAED2_scooby_doo_team

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Software engineering for ML systems

## Image classification of cats vs dogs 

This project contains all code necessary in order to create an api that exposes models trained for the purpose of image classificaion, more specifically, for cats vs dogs classification. The data has been selected from a subset of the Assirra dataset and the models selected were fine-tunned Vision Transformers. The specific model and data card can be found in the following links:

- <a target='_blank' href = 'https://github.com/taed2-2425q1-gced-upc/TAED2_scooby_doo_team/blob/master/docs/model_card.md'> Model Card </a>
- <a target = '_blank' href = 'https://github.com/taed2-2425q1-gced-upc/TAED2_scooby_doo_team/blob/master/docs/dataset_card.md' > Data Card </a>

## Project Organization

The project was originally built with cookiecutter's template, and then modified to suit our needs, the final organitzation is the following:

```

├── README.md                         <- The top level README
├── pyproject.toml                    <- Defines dependencies of python libraries
├── poetry.lock                       <- Defines which versions of python libraries to use
├── params.yaml                       <- Defines which models to use and which combinations of hyperparameters
│                                        to train
├── .*                                <- Other configuration files (such as .gitignore, .dvcignore, etc), not
│                                        stated for simplicity
├── data                              <- Folder containing the data to train and validate the model
│   ├── processed                     <- The final data processed for model use
│   │   ├── train_images              <- Images used during training
│   │   │   ├── cats                  <- Images of cats
│   │   │   └── dogs                  <- Images of dogs
│   │   ├── valid_images              <- Images used during validation
│   │   │   ├── cats                  <- Images of cats
│   │   │   └── dogs                  <- Images of dogs
│   │   └── *.csv                     <- Files generated during processing of data such as target class of
│   │                                    each image
│   └── raw                           <- The original, immutable data dump, stored in parquet files
├── docs                              <- Documents such as model card or data card
├── metrics                           <- Dumps of metrics such as the accuracy of the best model
├── reports                           <- Generated reports such as the ones generated through deepchecks
├── tests                             <- Tests to be used with pytest
├── models                            <- Compressed dumps of the trained models    
└── src                               <- Source code for use in this project
    ├── __init__.py                   <- Makes TAED2_scooby_doo_team a Python module
    ├── config.py                     <- Configuration file for python
    ├── features                      <- Files for feature extraction and data processing
    │   ├── deepchecks_validation.py  <- Performs data checks and writes a report in reports
    |   └── prepare.py                <- Creates the dataset for trainig
    ├── models                        <- Files for model training and evaluation
    |   ├── evaluate.py               <- Validation of the model
    |   ├── test.py                   <- Test the model
    |   └── train.py                  <- Train the model
    └── app                           <- Code to run the api
        └── api.py                    <- Api built in fastapi

```

## Instructions 

To run the code follow these instructions: 

**Clone the repository**

Clone the github repository by running the following command:

```bash
git clone https://github.com/taed2-2425q1-gced-upc/TAED2_scooby_doo_team.git
```

**Create virtual environemnt**

The first thing you will need is python3.10 or higher to create a virtual environment. There are several ways to do this but in our team we did the following steps. First you will need to create a virtual environment with the following command and run it.

```bash
python3 -m venv venv
source ./venv/bin/activate
```

Then you should install poetry inside this virtual environment, and install all other dependencies using the following command.

```bash
pip3 install poetry==1.8.2
poetry install
```

Since we already have a lock file in this repository this should install everything you will need. With this you have the virtual environment created.

**Get data and models**

To get the data and the models you will need access to the dagshub repository.

```bash
dvc pull
```

Alternatively, if you do not have access to the dagshub repository, you can run the dvc pipeline, which will create the processed data and will train some models which you will be able to use with the api.

```bash
dvc repro
```

**Run the api**

Now you simply need to run the api with the following command.

```bash
uvicorn src.app.api:app --host 0.0.0.0 --port 5000 --reload --reload-dir src/app --reload-dir models
```

## Tests

Reports of pytest where created by running the following command.

```bash
pytest --cov=src --cov-report=html:reports/coverage
```

## Contact

- Oriol Lopez Petit: oriol.lopez.petit@estudiantat.upc.edu
- Pablo Gete Alejandro: pablo.gete@estudiantat.upc.edu
- David Torrecilla Tolosa: david.torrecilla.tolosa@estudiantat.upc.edu
- Xiao Li Segarra: xiao.li.segarra@estudiantat.upc.edu 
- Xavier Pacheco Bach: xavier.pacheco.bach@estudiantat.upc.edu 

# Link to API

Our api can be found at: http://nattech.fib.upc.edu:40370/

--------

