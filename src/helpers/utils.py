import numpy as np
from sklearn import metrics
from scipy.spatial import distance

import perceval as pcvl


# utility functions
def periodically_continued(a, b):
    interval = b - a
    return lambda f: lambda x: f((x - a) % interval + a)


@periodically_continued(-1, 1)
def obs_param(x):
    return x

@periodically_continued(0, 2*np.pi)
def circ_param(x):
    return x


def bernoulli_delta(n_params, p=0.5):
    """
    Compute Bernoulli delta
    """
    delta_k = np.random.binomial(1, p, n_params)
    delta_k[delta_k == 0] = -1
    return delta_k


def gaussian_pdf(range):
    """
    Generate gaussian distribution function
    """
    x = np.arange(range)
    mu = x[range // 2]
    sigma = x[range // 8]

    pl = (
        1.0
        / np.sqrt(2 * np.pi * sigma ** 2)
        * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    )
    return pl / pl.sum()


def gaussian_mixture_pdf(range):
    """
    Generate mixed gaussian distribution function
    """
    x = np.arange(range)
    mu1 = x[2 * range // 7]
    mu2 = x[5 * range // 7]
    #sigma = x[range // 8]
    
    sigma1 = x[range // 8]
    #sigma2 = x[range // 16]
    sigma2 = x[range // 8]

    pl = (
        1.0
        / np.sqrt(2 * np.pi * sigma1 ** 2)
        * (np.exp(-((x - mu1) ** 2) / (2.0 * sigma1 ** 2)))
        + 1.0
        / np.sqrt(2 * np.pi * sigma2 ** 2)
        * (np.exp(-((x - mu2) ** 2) / (2.0 * sigma2 ** 2)))
    )
    return pl / pl.sum()


def state_to_int(state):
    """
    State to integer mapping
    """
    m = state.m
    res = 0
    for i in range(m):
        res += state[i] * (m + 1) ** (m - i)
    return res


def get_output_map(possible_states_list, bin_count, mapping_type = 'integer_mapping'):
    """
    Generate output map between possible output Fock states and target space (integers or grid). 
    """
    out_map = {}
    
    if mapping_type == 'integer_mapping':
        
        rev_map = {}
        possible_outputs = []
        
        for key in possible_states_list:
            int_state = state_to_int(key)
            rev_map[int_state] = key
            possible_outputs.append(int_state)
                
        new_index_vector = np.array_split(np.arange(len(possible_states_list)), bin_count)
        new_index_dict = {}
        for index_coarse, index_fine_list in enumerate(new_index_vector):
            for index_fine in index_fine_list:
                new_index_dict[index_fine] = index_coarse

        for index, int_state in enumerate(sorted(list(possible_outputs))):
            state = rev_map[int_state]
            out_map[state] = new_index_dict[index]
                    
    elif mapping_type == 'grid_mapping':
        
        for key in possible_states_list:
            out_map[key] = np.array(key)
    
    return out_map


def generate_possible_states(circuit, input_state, pnr = True):
    """
    Given a circuit, an input state and PNR information, get list of possible output states
    """
    temp_proc = pcvl.Processor("SLOS")
    temp_proc.set_circuit(circuit)
    temp_proc.with_input(input_state)
    temp_sampler = pcvl.algorithm.Sampler(temp_proc)
    
    if pnr:
        possible_states_list = list(temp_sampler.probs()["results"].keys())
    else:
        possible_states_list = [key for key in temp_sampler.probs()["results"].keys() if all(i < 2 for i in key)]
    
    return possible_states_list


def no_collision_to_threshold(circuit, input_state):
    """
    Given a circuit and an input state, create a mapping from no collision to threshold statistics
    """
    temp_proc = pcvl.Processor("SLOS")
    temp_proc.set_circuit(circuit)
    temp_proc.with_input(input_state)
    temp_sampler = pcvl.algorithm.Sampler(temp_proc)
    
    no_collision_states = [key for key in temp_sampler.probs()["results"].keys() if any(i > 1 for i in key)]
    
    no_collision_mapping = {}
    
    for state in no_collision_states:
        new_state = [i if i <= 1 else 1 for i in state]
        no_collision_mapping[state] = pcvl.BasicState(new_state)
    
    return no_collision_states, no_collision_mapping
    

class RBFMMD2(object):
    """
    MMD^2 with RBF (Gaussian) kernel.
    
    Args:
        sigma_list (list): a list of bandwidths.
        basis (1darray): defininng space.
      
    Attributes:
        K (2darray): full kernel matrix, notice the Hilbert is countable.
    """

    def __init__(self, sigma_list, basis):
        self.sigma_list = sigma_list
        self.basis = basis
        self.K = mix_rbf_kernel(basis, basis, self.sigma_list)

    def __call__(self, px, py):
        """
        Args:
            px (1darray, default=None): probability for data set x, used only when self.is_exact==True.
            py (1darray, default=None): same as px, but for data set y.

        Returns:
            float: loss.
        """
        pxy = px - py
        return self.kernel_expect(pxy, pxy)

    def kernel_expect(self, px, py):
        """
        expectation value of kernel function.
        
        Args:
            px (1darray): the first PDF.
            py (1darray): the second PDF.
            
        Returns:
            float: kernel expectation.
        """
        return px.dot(self.K).dot(py)


def mix_rbf_kernel(x, y, sigma_list):
    """
    multi-RBF kernel.
    
    Args:
        x (1darray|2darray): the collection of samples A.
        x (1darray|2darray): the collection of samples B.
        sigma_list (list): a list of bandwidths.
        
    Returns:
        2darray: kernel matrix.
    """
    ndim = x.ndim
    if ndim == 1:
        exponent = np.abs(x[:, None] - y[None, :]) ** 2
    elif ndim == 2:
        exponent = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    else:
        raise
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma)
        K = K + np.exp(-gamma * exponent)
    return K


class mmd_rbf(object):
    """Maximum Mean Discrepancy 
    Using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    
    Credit: https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    """
    def __init__(self, Y, gamma=1.0):
        self.bandwidth = gamma
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.YY = metrics.pairwise.rbf_kernel(Y, Y, self.bandwidth)
        print('init MMD loss')

    def __call__(self, X, Y):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        XX = metrics.pairwise.rbf_kernel(X, X, self.bandwidth)
        XY = metrics.pairwise.rbf_kernel(X, Y, self.bandwidth)
        return XX.mean() + self.YY.mean() - 2 * XY.mean()
    
    
def sample_from_target_pdf(target_space, n_target_samples, target_pdf):
    """
    Create n samples over a target space, given a distribution
    """
    return np.random.choice(target_space, n_target_samples, p=target_pdf)
    

class kl_divergence(object):
    """
    Compute KL divergence between two distributions
    """
    def __init__(self):
        pass
    
    def __call__(self, p, q):
        KL_div =  0
        for i in range(len(p)):
            if q[i]>0 and p[i]>0:
                KL_div += p[i] * np.log2(p[i]/q[i])
        return KL_div
    

def variance(data):
    """
    Given data, get variance.
    """
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return variance


def func1(x, a):
    """
    As implemented in photon recycling code
    """
    return a * np.exp(-median_of_means * x) + 1/comb(m,n)


class jensen_shannon_distance(object):
    """
    Using scipy function to get the Jensen-Shannon distance
    """
    def __init__(self):
        pass
    
    def __call__(self, p, q):
        return distance.jensenshannon(p, q)


class tvd(object):
    """
    Compute total variational distance between two distributions
    """
    def __init__(self):
        pass
    
    def __call__(self, p, q):
        tvd = 0.5*sum(abs(p - q))
        return tvd

    
    