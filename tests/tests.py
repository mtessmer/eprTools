import pytest, pickle
import numpy as np
import matplotlib.pyplot as plt
from eprTools import utils, DEERSpec

# Generate parameters and load test_data for tests
r = np.linspace(15, 100, 2 ** 10)
t = np.linspace(-1000, 4500, 2 ** 10)

ans_list = []
with np.load('test_data/gen_kern.npz') as f:
    for K in f.keys():
        ans_list.append(f[K])

kws = [{}, {'r': 80, 'time': 2500},
       {'r': r, 'time': 5000},
       {'size': 100},
       {'time': t, 'size': 100}]

test_data = zip(kws, ans_list)

class TestDEER:

    def test_from_file(self):
        spc = DEERSpec.from_file('test_data/Example_DEER.DTA')

        with open('test_data/from_file.pkl', 'rb') as f:
            spc_true = pickle.load(f)

        assert spc_true == spc

    @pytest.mark.parametrize('method', ['nnls', 'cvx'])
    def test_fit_method(self, method):
        spc = DEERSpec.from_file('test_data/Example_DEER.DTA')
        spc.get_fit(fit_method=method)

        assert spc.fit_method == method

        with open('test_data/fit_method.pkl', 'rb') as f:
            spc_true = pickle.load(f)

        assert spc == spc_true

    phi = np.linspace(0, 180, 18)
    V_ts=[]
    with np.load('test_data/set_phase.npz') as f:
        for i in range(len(f.files)):
            V_ts.append(f[f'arr_{i}'])
    print(V_ts)
    @pytest.mark.parametrize('phase, expected', zip(phi, V_ts))
    def test_set_phase(self, phase, expected):
        true_real, true_imag = expected
        spc = DEERSpec.from_file('test_data/Example_DEER.DTA')
        spc.set_phase(phase)
        np.testing.assert_almost_equal(true_real, spc.real)
        np.testing.assert_almost_equal(true_imag, spc.imag)


class TestUtils:

    @pytest.mark.parametrize('kwargs, expected', test_data)
    def test_generate_kernel(self, kwargs, expected):
        K = utils.generate_kernel(**kwargs)
        np.testing.assert_almost_equal(expected, K)

    def test_generate_kernel_nm(self):
        rnm = np.linspace(1.5, 10, 2**10)
        tnm = np.linspace(-1, 4.5, 2**10)

        Knm = utils.generate_kernel(rnm, tnm)
        K = utils.generate_kernel(r, t)

        np.testing.assert_almost_equal(K, Knm)

