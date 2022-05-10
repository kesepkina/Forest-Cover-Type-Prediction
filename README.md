# Forest-Cover-Type-Prediction
Homework for RS School Machine Learning course.

[Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset used.

## Usage
This package allows you to train model for classifying forest categories.
1. Clone this repository to your machine.
2. Download 
[Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train <ml model> -d <path to csv with data> -s <path to save trained model>
```
To get a full list of available models, use help:
```sh
poetry run train --help
```
You can configure additional options (such as hyperparameters and usage of nested cross-validation) in the CLI. To get a full list of them, use help:
```sh
poetry run train <ml model> --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
Below you can see screenshots with runs of different models with different hyperparameters and feature engineering techniques sorted by AUC score.

**KNN:**
![MLFlow knn](https://user-images.githubusercontent.com/43503787/166750880-e9d085b0-607f-4a88-be33-53a5d1c7e258.png)

**Logistic regression:**
![MLFlow logreg](https://user-images.githubusercontent.com/43503787/166762861-3c2818dc-6879-4814-9b4a-4daadeece4c7.png)

**Random forest:**
![MLFlow forest](https://user-images.githubusercontent.com/43503787/166759763-1aece55b-2f7d-4ddf-b06a-19fe449010b8.png)

**Using Nested-CV for hyperparameter tuning:**
...

## Development

The code in this repository must be tested, formatted with _black_, and pass _mypy_ typechecking before being commited to the repository.

Install all requirements (including dev requirements) to _poetry_ environment:
```
poetry install
```
Now you can use developer instruments, e.g. _pytest_:
```
poetry run pytest
```
*Expected result:*
![Tests](https://user-images.githubusercontent.com/43503787/167138592-4e36848f-4dd5-4c74-957f-b3f073186c4f.png)

To format my code automatically while commiting to git, I've added _pre-commit_, _black_ and _flake8_ libraries:
![Black and flake8](https://user-images.githubusercontent.com/43503787/167148781-672a0fef-3818-4030-8483-d4f4ceba91a5.png)

You can run _mypy_ to ensure the types are correct:
```
poetry run mypy <folder or file name>
```

_Expected result:_

<img src="https://user-images.githubusercontent.com/43503787/167158376-c5fd731d-a8af-49be-a427-5c6162dd5085.png" width="700">

More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
nox [-r]
```

_Expected result:_

<img src="https://user-images.githubusercontent.com/43503787/167690225-b8ee2377-377d-4c5f-8c25-68e28dd04744.png" width="500">
