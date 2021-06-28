from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation, get_performance_indicator, get_reference_directions
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.moead import MOEAD
from pymoo.algorithms.ctaea import CTAEA
from pymoo.optimize import minimize
from utils.sampling import *

import numpy as np
import pandas as pd

np.random.seed(16)

def compute_mean_HV (algorithm, problems, iterations):    #fixed problems, can pass later
    hypervolume_scores = np.zeros([len(problems), iterations])
    for i in range(iterations):
        results = []
        ind = 0
        for j in problems:
            problem = get_problem(j[0])
            res = minimize(problem, algorithm, ('n_gen', 5), seed=i, verbose=False)
            if res.F is not None:
                hv = get_performance_indicator("hv", ref_point=j[1]).calc(res.F)
            else:
                hv = 0
            results.append((j[0], res, res.F, hv))
            hypervolume_scores[ind, i] = hv
            ind = ind + 1
    mean_hv = np.mean(hypervolume_scores, axis = 1)
    std = np.std(hypervolume_scores, axis=1)
    output = []
    for i in range(len(problems)):
        output.append([problems[i][0], mean_hv[i], std[i]])
    path = 'results/' + algorithm.__module__ + '.csv'
    pd.DataFrame(np.array(output)).to_csv(path,
                                        header=('function', 'mean HV', 'std'),
                                       index=False)
    print("all runs:", hypervolume_scores)
    return(hypervolume_scores, print("Output written to " + path +'\n'))

problems = (('bnh', np.array([140, 50])),
            ('tnk', np.array([2, 2])),
            ('ctp1', np.array([1, 2])),
            ('zdt4', np.array([1,260])),
            ('kursawe', np.array([-10,2])),
            ('welded_beam', np.array([350, 0.1])),
            ('carside', np.array([42, 4.5, 13])))

ref_dir = get_reference_directions("das-dennis", 2, n_partitions=10)
algorithms = (NSGA2(pop_size=10), MOEAD(ref_dir, n_neighbors=10), CTAEA(ref_dir))
for algorithm in algorithms:
    print("Minimizing functions with " + algorithm.__module__)
    compute_mean_HV(algorithm, problems, 50)