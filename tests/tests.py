import pytest, pickle
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from eprTools import utils, DEERSpec
import cvxopt as cvo
from time import time
#import deerlab as dl
# Generate parameters and load test_data for tests
r = np.linspace(15, 100, 2 ** 10)
t = np.linspace(-1000, 4500, 2 ** 10)

kws = [{}, {'r': 80, 'time': 2500},
       {'r': r, 'time': 5000},
       {'size': 100},
       {'time': t, 'size': 100}]

ans_list = []
with np.load('test_data/gen_kern.npz') as f:
    for K in f.keys():
        ans_list.append(f[K])

test_data = zip(kws, ans_list)

class TestDEER:

    def test_from_file(self):
        spc = DEERSpec.from_file('test_data/Example_DEER.DTA')

        with open('test_data/from_file.pkl', 'rb') as f:
            spc_true = pickle.load(f)

        assert spc_true == spc

    def test_default(self):
        r = np.linspace(15, 100, 256)
        spc = DEERSpec.from_file('test_data/Example_DEER.DTA', r=r)
        spc.set_zero_time(-75)
        #t1 = time()
        spc.get_fit()
        # print('mht_fit_time: ', time() - t1)

        # t1 = time()
        # f = dl.fitsignal(spc.real, spc.time, spc.r)
        # print('dl_fit_time: ', time() - t1)
        #
        # fig, (ax1, ax2) = plt.subplots(2)
        # ax1.plot(spc.time, spc.real)
        # ax1.plot(spc.time, spc.fit)
        # ax1.plot(spc.time, spc.background - spc.lam)
        # ax1.plot(spc.time, f.V)
        # print(dir(f))
        # ax1.plot(spc.time, f.B - f.bgparam)
        # ax2.plot(spc.r, spc.P)
        # ax2.plot(spc.r, f.P)
        # plt.show()

        with open('test_data/default.pkl', 'rb') as f:
            spc_true = pickle.load(f)

        np.testing.assert_almost_equal(spc.fit, spc_true.fit)
        np.testing.assert_almost_equal(spc.P, spc_true.P)
        np.testing.assert_almost_equal(spc.K, spc_true.K)

    r = np.linspace(15, 100, 256)
    t = np.linspace(-500, 3500, 256)
    K, r, t = utils.generate_kernel(r, t)
    P = norm(30, 1).pdf(r) + 20 * norm(105, 5).pdf(r)
    P /= P.sum()
    S = K @ P
    lam = 0.15
    B = np.exp(-1e-5 * np.abs(t))
    V = (1 - lam + lam * S) * B + np.random.normal(0, 0.001, 256)

    def test_from_array(self, V):
        spc = DEERSpec.from_array(t, V, r)
        spc.save("test_data/from_array.pkl")
        with open("test_data/from_array.pkl", 'rb') as f:
            spc_true = pickle.load(f)

    def test_spnnls(self):
        r = np.linspace(15, 100, 256)
        spc = DEERSpec.from_file('test_data/Example_DEER.DTA', r=r)
        spc.set_zero_time(-75)
        spc.nnls = 'spnnls'
        spc.get_fit()

        with open('test_data/spnnls.pkl', 'rb') as f:
            spc_true = pickle.load(f)

        np.testing.assert_almost_equal(spc.fit, spc_true.fit)
        np.testing.assert_almost_equal(spc.P, spc_true.P)
        np.testing.assert_almost_equal(spc.K, spc_true.K)

    def test_extra_reg(self):
        spc = DEERSpec.from_array(t, V, r)

        def extra_reg_nnls(K, L, V, alpha, abstol=1e-9, reltol=1e-8):
            beta = 10
            M = np.zeros(len(r))
            M[r > 60] = 1
            M = np.diag(M)

            KtK = K.T @ K + alpha ** 2 * L.T @ L + beta ** 2 * M.T @ M
            KtV = - K.T @ V.T

            # get unconstrained solution as starting point.
            P = np.linalg.inv(KtK) @ KtV
            P = P.clip(min=0)

            B = cvo.matrix(KtK)

            A = cvo.matrix(KtV)

            # Minimize with CVXOPT constrained to P >= 0
            lower_bound = cvo.matrix(np.zeros_like(P))
            G = -cvo.matrix(np.eye(len(P), len(P)))
            cvo.solvers.options['show_progress'] = False
            cvo.solvers.options['abstol'] = abstol
            cvo.solvers.options['reltol'] = reltol
            fit_dict = cvo.solvers.qp(B, A, G, lower_bound, initvals=cvo.matrix(P))
            P = fit_dict['x']
            P = np.squeeze(np.asarray(P))

            return P

        spc.nnls = extra_reg_nnls
        spc.get_fit()

        with np.load('test_data/extra_reg.npz') as f:
            np.testing.assert_almost_equal(spc.fit, f['fit'])
            np.testing.assert_almost_equal(spc.P, f['P'])
            np.testing.assert_almost_equal(spc.K, f['K'])

    def test_from_distribution(self):
        r = np.linspace(1, 100, 256)
        P = norm(45, 3).pdf(r)
        P /= P.sum()
        time = 3500
        spc = DEERSpec.from_distribution(r, P, time)

        with open('test_data/from_distribution.pkl', 'rb') as f:
            spc_true = pickle.load(f)

        np.testing.assert_almost_equal(spc_true.spec, spc.spec)
        np.testing.assert_almost_equal(P, spc.P)

    @pytest.mark.parametrize('method', ['spnnls', 'cvxnnls'])
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
        true_spec = expected
        spc = DEERSpec.from_file('test_data/Example_DEER.DTA')
        spc.set_phase(phase)
        print(phase)

        np.testing.assert_almost_equal(true_spec, spc.spec)


class TestUtils:

    @pytest.mark.parametrize('kwargs, expected', test_data)
    def test_generate_kernel(self, kwargs, expected):
        K, r, t = utils.generate_kernel(**kwargs)
        np.testing.assert_almost_equal(expected, K)

    def test_generate_kernel_nm(self):
        rnm = np.linspace(1.5, 10, 2**10)
        tnm = np.linspace(-1, 4.5, 2**10)

        Knm, _, _ = utils.generate_kernel(rnm, tnm)
        K, _, _ = utils.generate_kernel(r, t)

        np.testing.assert_almost_equal(K, Knm)

