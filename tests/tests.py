import pytest, pickle
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from eprTools import utils, DEERSpec
from time import time
import deerlab as dl
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

        fig, ax = plt.subplots(2)
        ax[0].plot(spc.time[:50], spc.real[:50])
        ax[0].axvline(0)
        ax[1].plot(spc.raw_time[:50], spc.raw_real[:50])
        ax[1].axvline(0)
        plt.show()
        assert spc_true == spc

    def test_default(self):
        spc = DEERSpec.from_file('test_data/Example_DEER.DTA', r=(15, 60))
        t1 = time()
        spc.get_fit()
        print('mht_fit_time: ', time() - t1)

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(spc.time, spc.real)
        ax1.plot(spc.time, spc.fit)
        ax1.plot(spc.time, spc.background - spc.lam)

        ax2.plot(spc.r, spc.P)
        plt.show()

    def test_from_distribution(self):
        r = np.linspace(1, 100, 256)
        P = norm(45, 3).pdf(r)
        P /= P.sum()
        time = 3500
        spc = DEERSpec.from_distribution(r, P, time)
        spc.get_fit()

        plt.plot(r, P)
        plt.plot(spc.r, spc.P)
        plt.show()

        plt.plot(spc.time, spc.raw_spec_real)
        plt.plot(spc.time, spc.background)

        plt.show()

    def test_bg_corr(self):
        from eprTools import generate_kernel, DEERSpec
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        d = np.linspace(20, 120, 10)
        rin = np.linspace(20, 150, 512)
        r = np.linspace(20, 80, 512)
        t = np.linspace(-500, 6500, 512)
        Kin, r, t = generate_kernel(rin, t)
        k = 4e-5
        mod_depth = 0.08
        bg = np.exp(-k * np.abs(t))
        for mu in d:
            fig, ax = plt.subplots(1, 3, figsize=(30, 10.5))
            P = norm(mu, 4).pdf(rin)
            P /= P.sum()
            V_t = (1 - mod_depth + mod_depth * (Kin.dot(P))) * bg
            V_t /= V_t.max()
            spc = DEERSpec.from_array(time=t, spec_real=V_t, spec_imag=np.zeros(len(V_t)), r=(15, 100))
            spc.get_fit()
            print(spc.background_param)
            ax[0].plot(rin, P / P.sum())
            ax[0].plot(spc.r, spc.P / spc.P.sum())
            ax[0].legend(['True P(r)', 'Fit P(r)'])

            ax[1].plot(spc.time, spc.real)
            ax[1].plot(t, V_t)
            ax[1].plot(spc.fit_time, spc.fit)
            ax[1].plot(spc.time, spc.background)
            ax[1].legend(['spc vt', 'provided vt', 'fit vt', 'background'])

            ax[2].scatter(spc.rho, spc.eta)
            ax[2].scatter(spc.rho[spc.alpha_idx], spc.eta[spc.alpha_idx], color='r')
            plt.show()


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
        K, r, t = utils.generate_kernel(**kwargs)
        np.testing.assert_almost_equal(expected, K)

    def test_generate_kernel_nm(self):
        rnm = np.linspace(1.5, 10, 2**10)
        tnm = np.linspace(-1, 4.5, 2**10)

        Knm = utils.generate_kernel(rnm, tnm)
        K = utils.generate_kernel(r, t)

        np.testing.assert_almost_equal(K, Knm)

