import perceval as pcvl
import numpy as np
from math import comb
from copy import copy

from scipy.optimize import curve_fit


def check_no_collision(state) -> bool:
    return all(i <= 1 for i in state)


def handle_zero_photon_lost_dist(noisy_distributions, pattern_map, noisy_state, count):
    index = pattern_map[tuple(noisy_state)]
    noisy_distributions[0][index] += count


def handle_one_photon_lost_dist(noisy_distributions, pattern_map, noisy_state, count):
    for t in range(noisy_state.m):  # loop through each bit in string and +1 in each place
        n_ls = list(noisy_state)
        n_ls[t] += 1
        if check_no_collision(n_ls):
            index = pattern_map[tuple(n_ls)]
            noisy_distributions[1][index] += count


def handle_two_photons_lost_dist(noisy_distributions, pattern_map, noisy_state, count):
    for t in range(noisy_state.m):
        n_ls = list(noisy_state)
        n_ls[t] += 1

        for r in range(t, noisy_state.m):
            n_ls1 = copy(n_ls)
            n_ls1[r] += 1

            if check_no_collision(n_ls1):  # if non-collision is true
                index = pattern_map[tuple(n_ls1)]
                noisy_distributions[2][index] += count


def gen_lossy_dists(noisy_input, ideal_photon_count, pattern_map, threshold_stats = False):
    """
    Takes as input non-collision samples.
    Outputs an approximate distributions for each number of lost photon statistics.
    """
    max_lost_photons = 2
    noisy_distributions = [np.zeros(len(pattern_map)) for _ in range(max_lost_photons+1)]

    for noisy_state, count in noisy_input.items():  # loop through all the noisy states

        if threshold_stats:
            noisy_state = pcvl.BasicState([i if i <= 1 else 1 for i in noisy_state])
        else:
            noisy_state = noisy_state
        
        if noisy_state.n < (ideal_photon_count - max_lost_photons) or not check_no_collision(noisy_state):
            continue
        actual_photon_count = noisy_state.n

        if actual_photon_count == ideal_photon_count:
            handle_zero_photon_lost_dist(noisy_distributions, pattern_map, noisy_state, count)

        elif actual_photon_count == ideal_photon_count - 1:
            handle_one_photon_lost_dist(noisy_distributions, pattern_map, noisy_state, count)

        elif actual_photon_count == ideal_photon_count - 2:
            handle_two_photons_lost_dist(noisy_distributions, pattern_map, noisy_state, count)

    for i in range(max_lost_photons+1):
        noisy_distributions[i] = noisy_distributions[i]/sum(noisy_distributions[i])

    return noisy_distributions


def get_avg_exp(noisy_distributions, m, n):
    """
    Generates the exponent from average distance from uniform of lossy distributions.
    """

    exponent_list = []

    def func(x, a, b):
        return a * np.exp(-b * x) + 1/comb(m, n)

    for k in range(len(noisy_distributions[0])):
        z, _ = curve_fit(func, [0, 1, 2, 50], [noisy_distributions[0][k], noisy_distributions[1][k], noisy_distributions[2][k], 1/comb(m, n)],
                         bounds=([noisy_distributions[0][k], -5], [noisy_distributions[0][k]+0.00001, 5]), maxfev=2000000)
        exponent_list.append(z[1])

    y = [s for s in exponent_list if s < 4 and s > 0.001]
    p = y[:20]
    split_for_average = np.array(np.split(np.array(p), 5))
    split_means = split_for_average.mean(axis=1)
    median_of_means = np.median(np.array(split_means))

    return median_of_means


def get_avg_exp_from_uni_dist(noisy_distributions, m, n):

    def func(x, b):
        return uni_value * np.exp(-b * x)

    uniform_prob = 1/comb(m, n)
    noisy_distributions_from_uni = [np.average(abs(noisy_distribution - uniform_prob))
                                    for noisy_distribution in noisy_distributions]

    uni_value = noisy_distributions_from_uni[0]

    z, _ = curve_fit(func, [0, 1, 2, 50], [uni_value, noisy_distributions_from_uni[1],
                                           noisy_distributions_from_uni[2], 0], bounds=([-5], [5]), maxfev=2000000)

    return z


def standard_dev_decay_params(exponent_list):
    return np.std(exponent_list)
