import warnings
import time
import numpy as np
import pygmo as pg
from utils.sms_ego_functions import *
from utils.sampling import *


warnings.filterwarnings("ignore")


def E_SMS_EGO(problem, eval_budget, time_budget):
    dim = problem.n_var
    sample_size = 10*dim
    dimension_bounds = np.concatenate((problem.xl, problem.xu)).reshape(2, len(problem.xl)).T
    x = generate_MD_LHSample(dimension_bounds, dim, sample_size)
    y_values = problem.evaluate(x, return_values_of=["F", "feasible"])
    x = x[y_values[1].flatten()]
    y_values = y_values[0][y_values[1].flatten()]
    cv_splits = 10
    encountered = y_values
    predicted_y_gc = [] #  greatest contributor according to ensemble models
    counter = 1
    while counter < eval_budget+1 and time_budget > 0:
        start = time.time()
        models_per_objective, weights_per_objective, poi_per_objective = [], [], []
        for obj in range(problem.n_obj):
            print("\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A")
            print("Iteration:", counter, '/', eval_budget, "(Objective", obj+1,'/',problem.n_obj,')\n')
            output = minimize_objective(x, y_values.T[obj], cv_splits, dim, dimension_bounds)
            models_per_objective.append(output[0])
            weights_per_objective.append(output[1])
            poi_per_objective.append(output[2])  # points of interest
        poi_concatenated = np.concatenate(poi_per_objective)
        if poi_concatenated.size == 0:
            print("No new points found, optimization process complete.")
            return((x, y_values, (models_per_objective, weights_per_objective), predicted_y_gc))
        potential_points = np.zeros((problem.n_obj, len(poi_concatenated))).T  # obj * total points of interest matrix
        for i in range(len(models_per_objective)):
            for j in range(len(np.concatenate(poi_per_objective))):
                potential_points[j][i] = ensemble_prediction(poi_concatenated[j],
                                                             models_per_objective[i],
                                                             weights_per_objective[i],
                                                             x, y_values.T[i].reshape(-1, 1), dim)
        hv = pg.hypervolume(potential_points)
        encountered = np.concatenate((encountered, potential_points))
        reference_point = calc_rp(encountered)
        predicted_y_gc.append(potential_points[hv.greatest_contributor(reference_point)])
        greatest_contributor = [poi_concatenated[hv.greatest_contributor(reference_point)]]
        '''Evaluate new point greatest contributor'''
        evaluated_new = problem.evaluate(greatest_contributor, return_values_of=["F"])
        x = np.concatenate((x, [greatest_contributor[0]]))
        y_values = np.concatenate((y_values, evaluated_new))
        end = time.time()
        time_budget = time_budget - (end - start)
        counter = counter+1
    return ((x, y_values, (models_per_objective, weights_per_objective), predicted_y_gc))

