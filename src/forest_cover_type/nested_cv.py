import click
from numpy import logspace, mean, std
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV


def nested_val_score(estimator, X, y, scoring):
    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    # enumerate splits
    outer_results = {"acc": [], "f1": [], "roc_auc_ovr": []}
    best_score = 0
    scores = {}
    for train_ix, test_ix in cv_outer.split(X, y):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=41)
        # define search space
        space = dict()
        if isinstance(estimator, LogisticRegression):
            space["max_iter"] = [100, 200, 300, 400]
            space["C"] = logspace(-2, 2, num=5)
        elif isinstance(estimator, KNeighborsClassifier):
            space["n_neighbors"] = [3, 5, 7, 9, 11]
            space["weights"] = ("uniform", "distance")
        elif isinstance(estimator, RandomForestClassifier):
            space["n_estimators"] = [100, 300]
            space["criterion"] = ("entropy", "gini")
            space["max_depth"] = [3, 7, 12, None]
        # define search
        search = GridSearchCV(
            estimator, space, scoring=scoring, cv=cv_inner, refit=True
        )
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)
        # evaluate the model
        scores["acc"] = accuracy_score(y_test, yhat)
        scores["f1"] = f1_score(y_test, yhat, average="macro")
        scores["roc_auc_ovr"] = roc_auc_score(
            y_test, y_proba, multi_class="ovr"
        )
        # store the result
        outer_results["acc"].append(scores["acc"])
        outer_results["f1"].append(scores["f1"])
        outer_results["roc_auc_ovr"].append(scores["roc_auc_ovr"])
        # report progress
        click.echo(
            ">acc=%.3f, f1=%.3f, roc_auc_ovr=%.3f, est=%.3f, cfg=%s"
            % (
                scores["acc"],
                scores["f1"],
                scores["roc_auc_ovr"],
                result.best_score_,
                result.best_params_,
            )
        )
        # save the best model
        if scores[scoring] > best_score:
            best_score = scores[scoring]
            best_params = result.best_params_
    # summarize the estimated performance of the model
    click.echo(
        "Accuracy: mean - %.3f (std - %.3f)"
        % (mean(outer_results["acc"]), std(outer_results["acc"]))
    )
    click.echo(
        "F1 score: mean - %.3f (std - %.3f)"
        % (mean(outer_results["f1"]), std(outer_results["f1"]))
    )
    click.echo(
        "ROC-AUC ovr: mean - %.3f (std - %.3f)"
        % (
            mean(outer_results["roc_auc_ovr"]),
            std(outer_results["roc_auc_ovr"]),
        )
    )
    return best_params


# depricated
def nested_val_score_2(estimator, X, y, scoring):
    # define search space
    space = {}
    if isinstance(estimator, LogisticRegression):
        space["max_iter"] = [100, 200, 300, 400]
        space["C"] = logspace(-2, 2)
    elif isinstance(estimator, KNeighborsClassifier):
        space["n_neighbors"] = [3, 5, 7, 9, 11]
        space["weights"] = ("uniform", "distance")
    elif isinstance(estimator, RandomForestClassifier):
        space["n_estimators"] = [10, 100, 300]
        space["criterion"] = ("entropy", "gini")
        space["max_depth"] = [3, 5, 7, None]
    cv_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator, space, scoring=scoring, cv=cv_inner, refit=True
    )
    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=41)
    # execute the nested cross-validation
    cv_accuracy = cross_val_score(
        search, X, y, scoring="accuracy", cv=cv_outer, n_jobs=-1
    ).mean()
    cv_f1 = cross_val_score(
        search, X, y, scoring="f1_macro", cv=cv_outer, n_jobs=-1
    ).mean()
    cv_auc_ovr = cross_val_score(
        search, X, y, scoring="roc_auc_ovr", cv=cv_outer, n_jobs=-1
    ).mean()
    return cv_accuracy, cv_f1, cv_auc_ovr
