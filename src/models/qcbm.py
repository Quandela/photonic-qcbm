import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler
from typing import Callable

from ..helpers import ParametrizedQuantumCircuit, SPSA
from ..helpers.utils import bernoulli_delta, get_output_map, generate_possible_states, no_collision_to_threshold
from ..helpers.utils import RBFMMD2, jensen_shannon_distance, tvd
from ..photon_recycling import photon_recycling


class QCBM:
    """
    Quantum Circuit Born Machine class

    :param parametrized_circuit: A Perceval Circuit containing variational parameters.
    :param input_state: Input state.
    :param sample_count: Number of samples used per run of the circuit, can also be seen as the batch size.
    :param loss_parameter: Balanced photon loss in the whole system [0,1], 0 runs a perfect simulation where no photon
                           is lost.
    :param pnr: True if the detector have photon number resolution, False if they are threshold detectors.
    :param loss_fun: Loss function chosen for the training. See helpers.utils for examples: kl, tvd, etc.
    :param target_pdf: Target probability distribution that the model tries to learn
    :param target_space: Space where the target data is defined.
    :param use_samples_only: True if training with samples only, requires a loss function like mmd_rbf.
    :param target_samples: Target samples from the data that the model tries to learn. 
                           This should be used instead of target_pdf if use_samples_only = True
    :param use_photon_recycling: If True, photon recycling will be applied on the lossy outputs of the circuit.
    :param miti: If True, photon recycling will return the mitigated distribution. 
                 If False, it will return the postselected distribution.
    :param bin_count: Specify a bin count for the data that the model is generating. 
                      If not specified, bin_count will match the number of possible output Fock states of the circuit.
    """
    def __init__(self, parametrized_circuit: ParametrizedQuantumCircuit,
                 input_state: pcvl.BasicState,
                 sample_count: int = 10000,
                 loss_parameter: float = 0,
                 pnr: bool = True,
                 threshold_stats: bool = False,
                 loss_fun: Callable = None,
                 target_pdf=None,
                 target_space=None,
                 use_samples_only: bool = False,
                 target_samples=None,
                 use_photon_recycling: bool = False,
                 miti: bool = True,
                 bin_count: int = None):
        self.pqc = parametrized_circuit
        self.x0 = self.pqc.init_params()
        self.input_state = input_state
        
        # define states to be kept (also relevant for pnr=False lossless case)
        self.postselected_states = generate_possible_states(self.pqc.circuit, input_state, pnr)
        self.bin_count = bin_count or len(self.postselected_states)
        self.threshold_stats = threshold_stats
        if threshold_stats:
            self.no_collision_states, self.no_collision_mapping = no_collision_to_threshold(self.pqc.circuit, input_state)

        self.output_map = get_output_map(self.postselected_states, self.bin_count)
        self.loss_fun = loss_fun
        self.target_pdf = target_pdf
        self.target_space = target_space
        self.use_samples_only = use_samples_only
        self.target_samples = target_samples
        self.use_photon_recycling = use_photon_recycling
        self.miti = miti

        self.sample_count = sample_count
        self.loss_parameter = loss_parameter
        
        # additional metrics
        self.tvd_metric_class = tvd()
        self.mmd_metric_class = RBFMMD2(sigma_list=[0.25], basis = np.arange(self.bin_count))
        self.js_metric_class = jensen_shannon_distance()
        
        proc = pcvl.Processor("SLOS")
        # if we wish to add photon distinguishability:
        #proc = pcvl.Processor("SLOS", source=pcvl.Source(losses = 0.0, emission_probability=1,
        #                                                 multiphoton_component=0, indistinguishability=0.92))
        
        if loss_parameter > 0:
            proc = pcvl.Processor("SLOS", source=pcvl.Source(losses=self.loss_parameter))
            # if we wish to add photon distinguishability:
            #proc = pcvl.Processor("SLOS", source=pcvl.Source(losses = self.loss_parameter, emission_probability=1,
            #                                                 multiphoton_component=0, indistinguishability=0.92))
            
            # add postselection rule in order to take photon loss into account
            proc.min_detected_photons_filter(0)
            
        proc.set_circuit(self.pqc.circuit)
        proc.with_input(input_state)

        # here, 1 shot = 1 sample
        self.sampler = Sampler(proc, max_shots_per_call=sample_count)

    # get the output pdf of the current QCBM
    def pdf(self):
        self.sampler.clear_iterations()
        self.sampler.add_iteration_list([
            {
                "circuit_params": self.pqc.var_param_map
            },
        ])
        res = self.sampler.sample_count(self.sample_count)["results_list"][-1]["results"]        

        dist = np.zeros(self.bin_count)
        
        total_count = 0
        for key in res.keys():
            if key in self.postselected_states:
                dist[self.output_map[key]] += res[key]
                total_count += res[key]
        
        if not self.use_samples_only:
            dist = np.divide(dist, total_count)
        
        print('total count is ' + str(total_count))
        
        return dist
    
    
    # get raw results (used as input for photon recycling)
    def raw_results(self):
        self.sampler.clear_iterations()
        self.sampler.add_iteration_list([
            {
                "circuit_params": self.pqc.var_param_map
            },
        ])
        res = self.sampler.sample_count(self.sample_count)["results_list"][-1]["results"]
        
        print(f'total count is {sum(res.values())}')
        return res
    
    
    def pdf_without_photon_loss(self):
        self.sampler.clear_iterations()
        self.sampler.add_iteration_list([
            {
                "circuit_params": self.pqc.var_param_map
            },
        ])
        res = self.sampler.sample_count(self.sample_count)["results_list"][-1]["results"]

        dist = np.zeros(self.bin_count)
        for key in res.keys():
            dist[self.output_map[key]] += res[key]
        
        if not self.use_samples_only:
            dist = np.divide(dist, self.sample_count)
       
        return dist
    
    
    # utility function to speed up computations on the QPU
    # calculates the pdf at different parameter endpoints (used in pseudo gradient)
    def pdf_diff(self, pos, neg):
        self.pqc.update_var_params(pos)
        pos_params = self.pqc.var_param_map
        self.pqc.update_var_params(neg)
        neg_params = self.pqc.var_param_map

        self.sampler.clear_iterations()
        self.sampler.add_iteration_list([
            {
                "circuit_params": pos_params
            },
            {
                "circuit_params": neg_params
            },
        ])
        result_list = self.sampler.sample_count(self.sample_count)["results_list"]
        res_pos = result_list[-2]["results"]
        res_neg = result_list[-1]["results"]

        dist_pos = np.zeros(self.bin_count)
        dist_neg = np.zeros(self.bin_count)
        
        total_count_pos = 0
        for key in res_pos.keys():
            if key in self.postselected_states:
                dist_pos[self.output_map[key]] += res_pos[key]
                total_count_pos += res_pos[key]
        
        if not self.use_samples_only:
            dist_pos = np.divide(dist_pos, total_count_pos)

        total_count_neg = 0
        for key in res_neg.keys():
            if key in self.postselected_states:
                dist_neg[self.output_map[key]] += res_neg[key]
                total_count_neg += res_neg[key]
        
        if not self.use_samples_only:
            dist_neg = np.divide(dist_neg, total_count_neg)
                
        return dist_pos, dist_neg


    # same utility function for raw results
    def raw_res_diff(self, pos, neg):
        self.pqc.update_var_params(pos)
        pos_params = self.pqc.var_param_map
        self.pqc.update_var_params(neg)
        neg_params = self.pqc.var_param_map

        self.sampler.clear_iterations()
        self.sampler.add_iteration_list([
            {
                "circuit_params": pos_params
            },
            {
                "circuit_params": neg_params
            },
        ])
        result_list = self.sampler.sample_count(self.sample_count)["results_list"]
        res_pos = result_list[-2]["results"]
        res_neg = result_list[-1]["results"]
        return res_pos, res_neg


    # apply photon recycling on raw results
    def get_photon_recycling_pdf(self, raw_results):
        # call photon recycling function
        miti_distribution, post_distribution, numbered_dictionary = \
            photon_recycling(raw_results,
                             ideal_photon_count=self.input_state.n,
                             threshold_stats = self.threshold_stats)
        # apply output map now
        # reorder the distribution
        dist = np.zeros(self.bin_count)
        
        for key in numbered_dictionary.keys():
            ind = numbered_dictionary[key]
            state = pcvl.BasicState(key)
            if self.miti:
                dist[self.output_map[state]] += miti_distribution[ind]
            else:
                dist[self.output_map[state]] += post_distribution[ind]
        
        return dist
    
    
    # get the current loss value
    def get_loss(self, params=None):
        if params is not None:
            self.pqc.update_var_params(params)
            
        if self.use_photon_recycling:
            pdf = self.get_photon_recycling_pdf(self.raw_results())
        else:
            pdf = self.pdf()
        
        if self.use_samples_only:
            # send samples to the loss function
            exp_samples = np.repeat(self.target_space, pdf.astype(int))
            loss_value = self.loss_fun(exp_samples, self.target_samples)
        else:
            # send probability distributions to the loss function
            loss_value = self.loss_fun(pdf, self.target_pdf)
            
            # add other metrics
            metric_tvd = self.tvd_metric_class(pdf, self.target_pdf)
            metric_mmd = self.mmd_metric_class(pdf, self.target_pdf)
            metric_js = self.js_metric_class(pdf, self.target_pdf)
        
        return loss_value, metric_tvd, metric_mmd, metric_js

    
    # pseudo gradient vector for SPSA
    def pseudo_grad(self, params, c=1e-3):
        delta_k = bernoulli_delta(len(params))
        
        if self.use_photon_recycling:
            raw_res_pos, raw_res_neg = self.raw_res_diff(params + c * delta_k, params - c * delta_k)
            pdf_pos = self.get_photon_recycling_pdf(raw_res_pos)
            pdf_neg = self.get_photon_recycling_pdf(raw_res_neg)
        else:
            pdf_pos, pdf_neg = self.pdf_diff(params + c * delta_k, params - c * delta_k)

        
        if self.use_samples_only:
            # send samples to the loss function
            exp_samples_pos = np.repeat(self.target_space, pdf_pos.astype(int))
            exp_samples_neg = np.repeat(self.target_space, pdf_neg.astype(int))
            loss_pos = self.loss_fun(exp_samples_pos, self.target_samples)
            loss_neg = self.loss_fun(exp_samples_neg, self.target_samples)
        else:
            # send probability distributions to the loss function
            loss_pos = self.loss_fun(pdf_pos, self.target_pdf)
            loss_neg = self.loss_fun(pdf_neg, self.target_pdf)

        grads = []
        for i in range(len(params)):
            grads.append((loss_pos - loss_neg) / (2 * c * delta_k[i]))

        self.pqc.update_var_params(params)
        return np.array(grads)

    
    # train the model according to the SPSA optimisation parameters provided
    def fit(self, spsa_iter_num, opt_iter_num, silent = False):
        params_prog = []
        spsa_step_duration = spsa_iter_num // opt_iter_num
        
        params = list(self.pqc.var_param_map.values())
        opt = SPSA(params, self.pseudo_grad, spsa_iter_num)

        loss_prog = []
        tvd_prog = []
        mmd_prog = []
        js_prog = []
        for i in range(opt_iter_num):
            params = opt.step(spsa_step_duration)
            self.pqc.update_var_params(params)
            #loss = self.get_loss()
            loss, metric_tvd, metric_mmd, metric_js = self.get_loss()
            
            # log and display results
            loss_prog.append(loss)
            params_prog.append(params)
            
            # adding metrics during tests
            tvd_prog.append(metric_tvd)
            mmd_prog.append(metric_mmd)
            js_prog.append(metric_js)

            if not silent:
                print("it", i)
                print("loss", loss)

        return loss_prog, params_prog, tvd_prog, mmd_prog, js_prog
    
