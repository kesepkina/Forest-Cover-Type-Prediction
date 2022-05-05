from numpy import logspace
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

def nested_val_score(estimator, X, y, scoring):
    # define search space
    space = {}
    if isinstance(estimator['classifier'], LogisticRegression):
        space['max_iter'] = [100, 200, 300, 400]
        space['C'] = logspace(-2, 2)
    elif isinstance(estimator['classifier'], KNeighborsClassifier):
        space['n_neighbors'] = [3, 5, 7, 9, 11]
        space['weights'] = ('uniform', 'distance')
    elif isinstance(estimator['classifier'], RandomForestClassifier):
        space['n_estimators'] = [10, 100, 300]
        space['criterion'] = ('entropy', 'gini')
        space['max_depth'] = [3, 5, 7, None]
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(estimator, space, scoring='roc_auc_ovr', n_jobs=1, cv=cv_inner, refit=True)
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=41)
    # execute the nested cross-validation
    scores = cross_val_score(search, X, y, scoring='roc_auc_ovr', cv=cv_outer, n_jobs=-1)
    return scores