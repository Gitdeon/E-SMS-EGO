from E_sms_ego import E_SMS_EGO
from pymoo.factory import get_problem, get_performance_indicator
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
np.random.seed(110)

def E_SMS_EGO_mean_HV (problems, iterations):
    hypervolume_scores = np.zeros([len(problems), iterations])
    for i in range(4,iterations):
        global results
        results = []
        ind = 0
        for j in problems:
            problem = get_problem(j[0])
            res = E_SMS_EGO(problem, eval_budget=25, time_budget=2500)
            if np.array(res[3]) is not None:
                hv = get_performance_indicator("hv", ref_point=j[1]).calc(np.array(res[1]))
            else:
                hv = 0
            results.append((j[0], res[0].tolist(), res[1].tolist(), hv, j[1].tolist()))
            respath = 'results/E_SMS_EGO_run_' + str(i) + '.csv'
            pd.DataFrame(results).to_csv(respath,
                                                  header=('problem', 'x', 'y', 'hv', 'ref'),
                                                  index=False)
            hypervolume_scores[ind, i] = hv
            ind = ind + 1
    mean_hv = np.mean(hypervolume_scores, axis = 1)
    std = np.std(hypervolume_scores, axis=1)
    output = []
    for i in range(len(problems)):
        output.append([problems[i][0], mean_hv[i], std[i]])
    path = 'results/E_SMS_EGO_res.csv'
    pd.DataFrame(np.array(output)).to_csv(path,
                                        header=('function', 'mean HV', 'std'),
                                        index=False)
    print("all runs:", hypervolume_scores)
    return(print("Output written to " + path +'\n'))


problems = (('bnh', np.array([140, 50])),
            ('tnk', np.array([2, 2])),
            ('ctp1', np.array([1, 2])),
            ('zdt4', np.array([1, 260])),
            ('kursawe', np.array([-20,2])),
            ('welded_beam', np.array([350, 0.1])),
            ('carside', np.array([42, 4.5, 13])))

E_SMS_EGO_mean_HV(problems, 10)