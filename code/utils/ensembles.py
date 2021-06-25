from utils.models import *
from sklearn.model_selection import KFold
from pymoo.factory import get_reference_directions
import numpy as np


def build_ensemble(x, y, num_splits, error_metric):
    K_folds = KFold(n_splits=num_splits)
    cv_error_measure = []
    weights = get_reference_directions("das-dennis", 5, n_partitions=10)
    count = 0
    for train_index, test_index in K_folds.split(x):
        count = count + 1
        print(
            '\x1b[1A\x1b[2K' + "Calculating optimal ensemble weights... [", (count / K_folds.get_n_splits()) * 100,
            "% ]")
        trained_models = train_models(x[train_index], y[train_index])
        pred = predict_models(trained_models, x[test_index])
        ensemble_error_measure = []
        for j in weights:
            y_pred = np.sum((j * np.array(pred).T), axis=1)
            ensemble_error_measure.append(error_metric(y_pred, y[test_index]))
        cv_error_measure.append(ensemble_error_measure)
    mean_cv_ensembles = np.mean(cv_error_measure, axis=0)
    best_weights = weights[np.argmin(mean_cv_ensembles)]
    return best_weights
