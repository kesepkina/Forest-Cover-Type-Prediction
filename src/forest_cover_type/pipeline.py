from scipy import rand
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def create_pipeline(
    use_scaler: bool,
    clf: str,
    **params
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if clf == 'logreg':
        classifier = LogisticRegression(
                random_state=params['random_state'],
                max_iter=params['max_iter'],
                C=params['logreg_c']
            )
    elif clf == 'knn':
        classifier = KNeighborsClassifier(
            n_neighbors=params['n_neighbors'],
            weights=params['weights']
        )
    elif clf == 'forest':
        classifier = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            bootstrap=params['bootstrap'],
            random_state=params['random_state']
        )
    pipeline_steps.append(
        (
            "classifier",
            classifier,
        )
    )
    return Pipeline(steps=pipeline_steps)