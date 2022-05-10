from numpy import logspace
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def nested_val_score(estimator, X, y, scoring):
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
    cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator, space, scoring=scoring, cv=cv_inner, refit=True
    )
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=4, shuffle=True, random_state=41)
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
