import pytest
import numpy as np
from eprTools import utils

# Generate parameters and load data for tests
r = np.linspace(15, 100, 2 ** 10)
t = np.linspace(-1000, 4500, 2 ** 10)

ans_list = []
with np.load('data/gen_kern.npz') as f:
    for K in f.keys():
        ans_list.append(f[K])

kws = [{}, {'r': 80, 'time': 2500},
       {'r': r, 'time': 5000},
       {'size': 100},
       {'time': t, 'size': 100}]

test_data = zip(kws, ans_list)

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

