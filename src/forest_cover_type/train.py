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


@click.command()
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
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
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
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    features, target = get_dataset(
        dataset_path
    )
    # features_train, features_val, target_train, target_val = train_test_split(
    #     features,
    #     target,
    #     random_state,
    #     test_split_ratio
    # )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
        # pipeline.fit(features_train, target_train)
        # y_pred = pipeline.predict(features_val)
        # accuracy = accuracy_score(target_val, y_pred)
        cv_accuracy = cross_val_score(pipeline, features, target, scoring='accuracy').mean()
        cv_f1 = cross_val_score(pipeline, features, target, \
                                scoring='f1_micro').mean()
        cv_log_loss = cross_val_score(pipeline, features, target, scoring='neg_log_loss').mean()
        # f1 = f1_score(target_val, y_pred)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy_cv", cv_accuracy)
        mlflow.log_metric("f1_score_cv", cv_f1)
        mlflow.log_metric("Neg_log_loss_score_cv", cv_log_loss)
        click.echo(f"Accuracy (CV): {cv_accuracy}.")
        click.echo(f"F1 score (CV): {cv_f1}.")
        click.echo(f"Negative Log loss score (CV): {cv_log_loss}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")