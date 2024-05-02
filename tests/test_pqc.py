from ..src.helpers import ParametrizedQuantumCircuit

import pytest
from perceval.components import Circuit


@pytest.mark.parametrize("arch", [['var'],
                                  ['var', 'var', 'var']])

def test_pqc_building(arch):
    M = 12

    pqc = ParametrizedQuantumCircuit(m=M, arch=arch)
    assert pqc.m == M
    assert isinstance(pqc.circuit, Circuit)
    assert len(pqc.circuit._components) == len(arch)
    for i in range(len(arch)):
        assert pqc.circuit._components[0][1].name.startswith(arch[0])
    var_params = pqc.get_params()
    assert len(var_params) == M*(M-1)*arch.count('var')

