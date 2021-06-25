from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
from utils.sampling import generate_random_sample
from utils.models import *
from utils.ensembles import *
import numpy as np
import numpy_indexed as npi

def max_EI(models, weights, x, y, dim, dimension_bounds):
    num_iters = 25
    EI_x = np.zeros([num_iters, dim])
    EI_y = np.zeros([num_iters, 1])
    for i in range(num_iters):
        print('\x1b[1A\x1b[2K' + "Finding new point with maximum Expected Improvement... [",
              float(int(i / num_iters * 100) + 1), "% ]")
        new_x = generate_random_sample(dimension_bounds, dim, 1)
        res = minimize(ensemble_prediction, x0=new_x, args=(models, weights, x, y, dim),
                       bounds=dimension_bounds)
        EI_x[i] = res.x
        EI_y[i] = res.fun - min(y)
    best_EI = EI_x[np.unique(EI_y, return_index=1)[1]]
    for i in best_EI:
        if i in x:  # check if already seen
            best_EI = np.delete(best_EI, np.where(best_EI == i)[0][0], axis=0)
    if best_EI.size == 0:
        print("No new point found for this objective, picking random point for exploration...")
        return (generate_random_sample(dimension_bounds, dim, 1))
    return (best_EI)


def minimize_objective(x, y, cv_splits, dim, dimension_bounds):  # Possible to leave out and integrate in Max_EI
    temp = npi.unique(x, return_index=1)
    x = temp[0]; y = y[temp[1]]
    model_weights = build_ensemble(x, y, cv_splits, sklearn.metrics.mean_absolute_error)
    trained_models = train_models(x, y)
    new_points_prediction = max_EI(trained_models, model_weights, x, y.reshape(-1,1), dim, dimension_bounds)
    return ((trained_models, model_weights, new_points_prediction))


def calc_rp(y_values):  # Calculate reference point
    reference_point = []
    for values in y_values.T:
        reference_point = np.append(reference_point, max(values))
    return (reference_point)

def ensemble_prediction(new_x, models, weights, x, y, dim):  # predicts minimal f(x) - stdev between models
    model_pred = []
    for model in models:
        if smtmodels.__name__ in getattr(model, '__module__', None):
            model_pred.append(model.predict_values(new_x.reshape(-1, dim)))
        elif 'sklearn' in getattr(model, '__module__', None) or 'pyearth' in getattr(model, '__module__', None):
            model_pred.append(model.predict(new_x.reshape(-1, dim)))
    pred = np.array(weights * np.array(model_pred).T)
    pv = np.sum(pred)
    knn_var = knnUncertainty(6, [new_x], pv, x, y)
    return pv - knn_var

def knnUncertainty(k, new_x, pred, x, y):
    '''
    Written by Bas van Stein, liacs Leiden (Modified by Gideon Hanse)
    '''
    # The measure of how certain a give prediction is given its k neighbours
    # k is the number of neighbours taken into account
    # new_x is the predicted point
    # pred contains the ensemble prediction of new_x
    # x is the set of known points (input)
    # y is the set of known points (output)
    no = MinMaxScaler(copy=True)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(no.fit_transform(x))
    new_x = no.transform(new_x)
    distances, indices = nbrs.kneighbors(new_x, k)
    dist = distances.flatten()
    ind = indices.flatten()
    # calculate the neirest point error
    abs_err = np.abs(pred - y[ind])
    weights = 1 - (dist / dist.sum())
    weighted_err = np.average(abs_err, weights=weights ** k, axis=0)
    nbrs_y = list(y[ind])
    nbrs_y.append(pred)
    nbrs_var = np.std(nbrs_y)
    min_dist = np.min(dist)
    sigma = weighted_err + min_dist * nbrs_var
    return(sigma)