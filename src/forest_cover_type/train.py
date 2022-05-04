from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, log_loss, f1_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

from .data import get_dataset, get_train_test_splits
from .pipeline import create_pipeline

def train_model(
    dataset_path: Path,
    save_model_path: Path,
    use_scaler: bool,
    clf: str,
    **model_params
) -> None:
    features, target = get_dataset(
        dataset_path
    )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, clf, **model_params)
        cv_accuracy = cross_val_score(pipeline, features, target, scoring='accuracy').mean()
        cv_f1 = cross_val_score(pipeline, features, target, \
                                scoring='f1_micro').mean()
        cv_log_loss = cross_val_score(pipeline, features, target, scoring='neg_log_loss').mean()
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy_cv", cv_accuracy)
        mlflow.log_metric("f1_score_cv", cv_f1)
        mlflow.log_metric("Neg_log_loss_score_cv", cv_log_loss)
        click.echo(f"Accuracy (CV): {cv_accuracy}.")
        click.echo(f"F1 score (CV): {cv_f1}.")
        click.echo(f"Negative Log loss score (CV): {cv_log_loss}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
        return

@click.group()
def train():
    pass

@train.command('logreg')
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=300,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
def train_logreg(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    model_params={'random_state': random_state, 'max_iter': max_iter, 'logreg_c': logreg_c}
    train_model(dataset_path, save_model_path, use_scaler, 'logreg', **model_params)


@train.command('knn')
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--n-neighbors",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--weights",
    default="uniform",
    type=click.Choice(['uniform', 'distance']),
    show_default=True,
)
def train_knn(
    dataset_path: Path,
    save_model_path: Path,
    use_scaler: bool,
    n_neighbors: int,
    weights: str,
) -> None:
    model_params={'n_neighbors': n_neighbors, 'weights': weights}
    train_model(dataset_path=dataset_path, save_model_path=save_model_path, use_scaler=use_scaler,
            clf='knn', **model_params)

@train.command('forest')
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default='gini',
    type=click.Choice(['gini', 'entropy']),
    show_default=True,
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    show_default=True,
)
@click.option(
    "--bootstrap",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
def train_forest(
    dataset_path: Path,
    save_model_path: Path,
    use_scaler: bool,
    n_estimators: int,
    criterion: str,
    max_depth: int,
    bootstrap: bool,
    random_state: int
) -> None:
    model_params={'n_estimators': n_estimators, 'criterion': criterion, 'max_depth': max_depth,
                    'bootstrap': bootstrap, 'random_state': random_state}
    train_model(dataset_path, save_model_path, use_scaler, 'forest', **model_params)