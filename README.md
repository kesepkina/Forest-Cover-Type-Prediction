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
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
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
