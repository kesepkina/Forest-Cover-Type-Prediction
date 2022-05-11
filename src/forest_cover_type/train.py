from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .data import get_dataset
from .pipeline import create_pipeline
from .nested_cv import nested_val_score


def train_model(
    dataset_path: Path,
    save_model_path: Path,
    scaler: str,
    feature_engineering: str,
    nested_cv: bool,
    clf: str,
    **model_params,
) -> None:
    features, target = get_dataset(dataset_path)
    with mlflow.start_run():
        if feature_engineering == "tsne":
            if scaler == "minmax_scaler":
                sc = MinMaxScaler()
            elif scaler == "st_scaler":
                sc = StandardScaler()
            features = sc.fit_transform(features)
            features = TSNE(
                init="pca", learning_rate="auto", random_state=42
            ).fit_transform(features)
            pipeline = create_pipeline(clf=clf, **model_params)
        else:
            pipeline = create_pipeline(
                clf, scaler, feature_engineering, **model_params
            )
        if nested_cv:
            if "scaler" in pipeline.named_steps.keys():
                features = pipeline["scaler"].fit_transform(features)
            if "feature_eng" in pipeline.named_steps.keys():
                features = pipeline["feature_eng"].fit_transform(features)
            model_params = nested_val_score(
                pipeline["classifier"], features, target, scoring="roc_auc_ovr"
            )
            pipeline["classifier"].set_params(**model_params)
        cv_accuracy = cross_val_score(
            pipeline, features, target, scoring="accuracy"
        ).mean()
        cv_f1 = cross_val_score(
            pipeline, features, target, scoring="f1_macro"
        ).mean()
        cv_auc_ovr = cross_val_score(
            pipeline, features, target, scoring="roc_auc_ovr"
        ).mean()
        mlflow.log_param("model", clf)
        mlflow.sklearn.log_model(pipeline["classifier"], clf)
        mlflow.log_param("scaler", scaler)
        mlflow.log_param("nested_cv", nested_cv)
        mlflow.log_param("Feature engineering type", feature_engineering)
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy_cv", cv_accuracy)
        mlflow.log_metric("f1_score_cv", cv_f1)
        mlflow.log_metric("AUC_ovr_cv", cv_auc_ovr)
        click.echo(f"Accuracy (CV): {cv_accuracy}.")
        click.echo(f"F1 score (CV): {cv_f1}.")
        click.echo(f"AUC One-vs-rest (CV): {cv_auc_ovr}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
        return


_common_options = [
    click.option(
        "-d",
        "--dataset-path",
        default="data/train.csv",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        show_default=True,
    ),
    click.option(
        "-s",
        "--save-model-path",
        default="data/model.joblib",
        type=click.Path(dir_okay=False, writable=True, path_type=Path),
        show_default=True,
    ),
    click.option(
        "--scaler",
        default="st_scaler",
        type=click.Choice(["st_scaler", "minmax_scaler"]),
        show_default=True,
    ),
    click.option(
        "-f",
        "--feature-engineering",
        default=None,
        type=click.Choice(["pca2", "pca3", "tsne"]),
        show_default=True,
        help="pca2 means pca with 2 components.",
    ),
    click.option(
        "--nested-cv/--no-nested-cv",
        default=False,
        show_default=True,
    ),
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@click.group()
def train(**kwargs):
    pass


@train.command("logreg")
@add_options(_common_options)
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
def train_logreg(
    dataset_path: Path,
    save_model_path: Path,
    scaler: str,
    feature_engineering: str,
    nested_cv: bool,
    random_state: int,
    max_iter: int,
    logreg_c: float,
) -> None:
    model_params = {
        "random_state": random_state,
        "max_iter": max_iter,
        "logreg_c": logreg_c,
    }
    train_model(
        dataset_path,
        save_model_path,
        scaler,
        feature_engineering,
        nested_cv,
        "logreg",
        **model_params,
    )


@train.command("knn")
@add_options(_common_options)
@click.option(
    "--n-neighbors",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--weights",
    default="uniform",
    type=click.Choice(["uniform", "distance"]),
    show_default=True,
)
def train_knn(
    dataset_path: Path,
    save_model_path: Path,
    scaler: str,
    feature_engineering: str,
    nested_cv: bool,
    n_neighbors: int,
    weights: str,
) -> None:
    model_params = {"n_neighbors": n_neighbors, "weights": weights}
    train_model(
        dataset_path,
        save_model_path,
        scaler,
        feature_engineering,
        nested_cv,
        "knn",
        **model_params,
    )


@train.command("forest")
@add_options(_common_options)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default="gini",
    type=click.Choice(["gini", "entropy"]),
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
    scaler: str,
    feature_engineering: str,
    nested_cv: bool,
    n_estimators: int,
    criterion: str,
    max_depth: int,
    bootstrap: bool,
    random_state: int,
) -> None:
    model_params = {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
        "bootstrap": bootstrap,
        "random_state": random_state,
    }
    train_model(
        dataset_path,
        save_model_path,
        scaler,
        feature_engineering,
        nested_cv,
        "forest",
        **model_params,
    )
