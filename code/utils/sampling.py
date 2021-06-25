import numpy as np
from utils.latin_hypercube_sampling import lhs

def scaleRescale(xStart, originalL, originalU, newlower, newupper):
    return (newupper - newlower) * ((xStart - originalL) / (originalU - originalL)) + newlower

def generate_MD_LHSample(bounds, num_dimensions, sample_size):
    '''
    Generate a multi dimensional sample with Latin Hypercube Sampling

    :param bounds: Tuple of lower and upper bounds per dimension (2 X No. of dimensions)
    :param sample_size: integer denoting sample size
    :return: LHS sample of multiple dimensions (No. of dimensions X sample size)
    '''
    sample = lhs(num_dimensions, sample_size).T
    for i in range(num_dimensions):
        sample[i] = scaleRescale(sample[i], 0, 1, bounds[i][0], bounds[i][1])
    return (np.array(sample).T)

def generate_random_sample(bounds, dim, sample_size=15):
    sample = np.random.random([dim, sample_size])
    for i in range(dim):
        sample[i] = scaleRescale(sample[i], 0, 1, bounds[i][0], bounds[i][1])
    return (np.array(sample).T)

