import re

import numpy as np
from typing import List

from perceval.utils import P
from perceval.components import PS, BS, Circuit, GenericInterferometer, catalog


class ParametrizedQuantumCircuit:
    """
    A parametrized quantum circuit with variational layers, to be used in the QCBM

    :param m: Number of modes.
    :param arch: Architecture of the circuit, i.e. how many times a variational block will be repeated.
    :param same_params_in_var: If True, variational blocks are repeated with the same parameters between blocks.
    :param one_param_per_interferometer: If True, there is only one parameter per MZI in the Generic Interferometer,
                                         and the other one is set to 0.
    """
    def __init__(self, m: int,
                 arch: List[str] = ["var"],
                 same_params_in_var: bool = False,
                 one_param_per_interferometer: bool = False):
        self.m = m
        self.arch = arch
        self.one_param_per_interferometer = one_param_per_interferometer
        self.same_params_in_var = same_params_in_var
        self.circuit = self.get_circuit()
        
        self.var_params = self.get_params()
        self.var_param_names = [p.name for p in self.var_params]

    def get_variational_clements_layer(self, layer_id):
        modes = self.m
        
        var_clem = Circuit(modes, name='var_clem_' + str(layer_id))
        mzi_generator_func = catalog['mzi phase last'].generate
        var_clem.add(0, GenericInterferometer(modes, mzi_generator_func))

        for param in var_clem.get_parameters():
            old_name = param.name
            param.name = old_name + '_' + str(layer_id + 1)
            if self.one_param_per_interferometer and 'b' in old_name:
                param.fix_value(0)
        
        return var_clem

    
    def get_circuit(self):
        mode_range = tuple([val.item() for val in np.arange(self.m, dtype=int)])
        active_modes = mode_range
        arch = self.arch
        c = Circuit(self.m)

        var_layer_num = 0
        clements_fixed_var_layer = None
        for layer_name in arch:
            split = re.split("\[|\]", layer_name)
            if len(split) == 1:
                layer_type = split[0]
            else:
                layer_type, modes_type = split[:-1]
                if ":" in modes_type:
                    start, end = modes_type.split(":")
                    active_modes = np.arange(int(start), int(end))
                else:
                    active_modes = np.array(modes_type.split(","), dtype=int)
                active_modes = tuple([val.item() for val in active_modes])

            if layer_type == "var":
                if self.same_params_in_var:
                    if clements_fixed_var_layer is None:
                        clements_fixed_var_layer = self.get_variational_clements_layer(var_layer_num)
                    c.add(mode_range, clements_fixed_var_layer)
                else:
                    c.add(mode_range, self.get_variational_clements_layer(var_layer_num))
                    var_layer_num += 1
            else:
                print('Unknown layer added')

        return c

    
    def get_params(self):
        params = self.circuit.get_parameters()
        var_params = []
        for p in params:
            var_params.append(p)
        return var_params

    
    def init_params(self, red_factor: float = 1, init_var_params=None):
        if init_var_params is None:
            var_param_map = self.update_var_params(np.random.normal(0, 2 * red_factor * np.pi, len(self.var_param_names)))
        else:
            var_param_map = self.update_var_params(init_var_params)
            
        for var_p in self.var_params:
            var_p.set_value(var_param_map[var_p.name])

        return list(self.var_param_map.values())

    
    def update_var_params(self, updated):
        updated_dict = {}
        for i, p in enumerate(self.var_params):
            new_val = updated[i]
            updated_dict[p.name] = new_val
            p.set_value(new_val)
        self.var_param_map = updated_dict
        return updated_dict

