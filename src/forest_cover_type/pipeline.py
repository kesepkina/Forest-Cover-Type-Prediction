from xml.sax.handler import feature_external_ges
from scipy import rand
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def create_pipeline(
    scaler: str,
    feature_eng: str,
    clf: str,
    **params
) -> Pipeline:
    pipeline_steps = []
    if scaler == 'st_scaler':
        sc = StandardScaler()
    elif scaler == 'minmax_scaler':
        sc = MinMaxScaler()
    if scaler is not None:
        pipeline_steps.append(('scaler', sc))

    if feature_eng == 'pca2':
        preprocessor = PCA(2)
        pipeline_steps.append(("feature_eng", preprocessor))
    elif feature_eng == 'pca3':
        preprocessor = PCA(3)
        pipeline_steps.append(("feature_eng", preprocessor))
    # if feature_eng is not None:
    #     pipeline_steps.append(("feature_eng", preprocessor))

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