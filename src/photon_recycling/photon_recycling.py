import time
import itertools
import numpy as np
from math import comb
from scipy.optimize import curve_fit

from perceval import BSCount

from .utils import gen_lossy_dists, get_avg_exp, get_avg_exp_from_uni_dist


def generate_one_photon_per_mode_mapping(m, n):
    combos = itertools.combinations(range(m), m - n)
    ones_photons = [1] * n
    char_cyc = itertools.cycle(ones_photons)
    perms = [tuple(0 if i in combo else next(char_cyc) for i in range(m))
             for combo in combos]
    return {perm: index for perm, index in zip(perms, range(len(perms)))}


def photon_recycling(noisy_input: BSCount, ideal_photon_count: int, threshold_stats = False):
    m = next(iter(noisy_input)).m
    pattern_map = generate_one_photon_per_mode_mapping(m, ideal_photon_count)
    noisy_distributions = gen_lossy_dists(noisy_input, ideal_photon_count, pattern_map, threshold_stats)

    # GENERATES THE AVERAGE EXPONENT USED FOR EXTRAPOLATION
    # median_of_means = get_avg_exp(noisy_distributions, m, ideal_photon_count)  # overwritten afterwards
    # print("median_of_means = ",median_of_means)

    # GET AVERAGE EXPONENT USING AVERAGE DISTANCE FROM UNIFORM PROBABILITY
    z = get_avg_exp_from_uni_dist(noisy_distributions, m, ideal_photon_count)
    median_of_means = z[0]

    # Generating the mitigated distribution using the decay parameter.
    mitigated_distribution = []
    c_mn_inv = 1 / comb(m, ideal_photon_count)

    def func1(x, a):
        return a * np.exp(-median_of_means * x) + c_mn_inv

    # print("\n Now generate mitigated distributions")

    for k in range(len(noisy_distributions[0])):
        z, _ = curve_fit(func1,
                         [1, 2, 50],
                         [noisy_distributions[1][k], noisy_distributions[2][k], c_mn_inv],
                         bounds=([-5], [5]),
                         maxfev=2000000)
        if noisy_distributions[1][k] > c_mn_inv > noisy_distributions[2][k]:
            mitigated_distribution.append(c_mn_inv)
        elif noisy_distributions[1][k] < c_mn_inv < noisy_distributions[2][k]:
            mitigated_distribution.append(c_mn_inv)
        else:
            mitigated_distribution.append(func1(0, z[0]))

    mitigated_distribution = [0 if i < 0 else i for i in mitigated_distribution]
    mitigated_distribution = mitigated_distribution / np.sum(mitigated_distribution)

    post_distribution = noisy_distributions[0]
    return mitigated_distribution, post_distribution, pattern_map
