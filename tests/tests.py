import pytest, pickle
from glob import glob
import numpy as np
from scipy.stats import norm, cauchy
import matplotlib.pyplot as plt
import cvxopt as cvo
from time import time
from eprTools import CWSpec, DeerExp, utils
from eprTools.nnls_funcs import NNLS_FUNCS

# Generate parameters and load test_data for tests
r = np.linspace(15, 100, 2 ** 10)
t = np.linspace(-1000, 4500, 2 ** 10)

kws = [{}, {'r': 80, 'time': 2500},
       {'r': r, 'time': 5000},
       {'size': 100},
       {'time': t, 'size': 100}]

ans_list = []
with np.load('test_data/gen_kern.npz') as fo:
    for K in fo.keys():
        ans_list.append(fo[K])

test_data = zip(kws, ans_list)
DTAFiles = glob('test_data/DEER_Data/*.DTA')

class TestDEER:

    def test_from_file(self):
        ex = DeerExp.from_file('test_data/Example_DEER.DTA')
        ex_ans = np.load('test_data/Example_DEER.npy')
        np.testing.assert_almost_equal(ex.V, ex_ans)

    def test_from_file_2D(self):
        ex = DeerExp.from_file('test_data/Example_DEER_2D.DTA')
        ex_ans = np.load('test_data/Example_DEER_2D.npy')
        np.testing.assert_almost_equal(ex.V, ex_ans)

    def test_default(self):
        r = np.linspace(15, 80, 256)
        ex = DeerExp.from_file('test_data/Example_DEER_2D.DTA', r=r)
        ex.get_fit()

        with np.load('test_data/default.npz', 'rb') as f:
            true_fit = f['fit']
            true_P = f['P']
            true_std = f['std']

        np.testing.assert_almost_equal(ex.fit, true_fit)
        np.testing.assert_almost_equal(ex.P, true_P)
        np.testing.assert_almost_equal(ex.std, true_std)

    def test_from_array(self):
        t = np.linspace(-100, 5000, 256)
        r = np.linspace(15, 80, 256)
        P = norm(35, 0.2).pdf(r)

        K, *_ = utils.generate_kernel(r, t)
        Kbg = (1 - 0.5 + 0.5 * K) * np.exp(-1e-3 * t)[:, None]
        V = Kbg @ P + np.random.normal(0, 0.02)

        ex = DeerExp.from_array(t, V, r)

        np.testing.assert_almost_equal(ex.real, V /V.max())

    @pytest.mark.parametrize('method, ans', zip(['cvxnnls', 'spnnls'], [0, 0]))
    def test_nnls(self, method, ans):
        t1 = time()
        r = np.linspace(15, 100, 256)
        ex = DeerExp.from_file('test_data/Example_DEER.DTA', r=r)
        ex.set_trim(3000)
        ex.nnls = method
        ex.get_fit()
        print(method, ': ', time() - t1)

    def test_extra_reg(self):
        t = np.linspace(-100, 5000, 256)
        r = np.linspace(15, 80, 256)
        K, _, _ = utils.generate_kernel(r, t)
        Kexp = (1 - 0.5 + 0.5 * K) * np.exp(-0.05 * np.abs(t))[:, None]
        P = norm(35, 2).pdf(r) + 0.5 * norm(90, 8).pdf(r)
        P /= np.trapz(P, r)
        V = K @ P

        ex = DeerExp.from_array(t, V, r)

        def extra_reg_nnls(K, L, V, alpha, abstol=1e-9, reltol=1e-8):
            beta = 50
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

        ex.nnls = extra_reg_nnls
        ex.get_fit()

        with np.load('test_data/extra_reg.npz') as f:
            np.testing.assert_almost_equal(ex.fit, f['fit'])
            np.testing.assert_almost_equal(ex.P, f['P'])
            np.testing.assert_almost_equal(ex.K, f['K'])

    def test_from_distribution(self):
        rr = np.linspace(1, 100, 256)
        P = norm(45, 3).pdf(rr)
        P /= P.sum()
        tt = -0.05 * 3500, 3500
        ex = DeerExp.from_distribution(rr, P, tt)

        with np.load('test_data/from_distribution.npz', 'rb') as f:
            true_V = f['V']
            true_P = f['P']


        np.testing.assert_almost_equal(ex.V, true_V)
        np.testing.assert_almost_equal(ex.P, true_P)

    @pytest.mark.parametrize('method', ['spnnls', 'cvxnnls'])
    def test_fit_method(self, method):
        ex = DeerExp.from_file('test_data/Example_DEER.DTA')
        ex.nnls = method
        assert ex.nnls == NNLS_FUNCS[method]

    @pytest.mark.parametrize('phi0', np.linspace(0, 170, 18))
    def test_set_phase(self, phi0):
        ex = DeerExp.from_file('test_data/Example_DEER.DTA')

        ex.set_phase(phi0)
        phase0 = ex.V.copy()

        ex.set_phase(phi0 + 90)
        phase90 = ex.V.copy()

        np.testing.assert_almost_equal(phase0.real, phase90.imag, decimal=3)

    def test_ci(self):
        ex = DeerExp.from_file('test_data/Example_DEER.DTA')
        ex.set_trim(3000)
        ex.get_fit()

        fig, ax =plt.subplots(2)
        ax[0].plot(ex.time, ex.real)
        ax[0].plot(ex.time, ex.fit)
        ax[1].plot(ex.r, ex.P)
        ax[1].fill_between(ex.r, *ex.ci(50), alpha=0.5)
        ax[1].fill_between(ex.r, *ex.ci(95), alpha=0.2)
        plt.show()

class TestCWSpec:

    def test_from_file_dta(self):
        spc = CWSpec.from_file('test_data/Example_CW.DTA', preprocess=True)

        with np.load('test_data/from_file.npz') as f:
            field_ans, spec_ans = f['field'], f['spec']

        np.testing.assert_almost_equal(spc.field, field_ans)
        np.testing.assert_almost_equal(spc.spec, spec_ans)

    def test_from_file_spc(self):
        spc = CWSpec.from_file('test_data/Example_CW.spc', preprocess=True)

        with np.load('test_data/from_file_spc.npz') as f:
            field_ans, spec_ans = f['field'], f['spec']

        plt.show()

        np.testing.assert_almost_equal(spc.field, field_ans)
        np.testing.assert_almost_equal(spc.spec, spec_ans)

    @pytest.mark.parametrize('ext', ['txt', 'dat', 'csv'])
    def test_from_file_ext(self, ext):
        spc = CWSpec.from_file(f'test_data/Example_CW.{ext}', preprocess=True)

        with np.load('test_data/from_file.npz') as f:
            field_ans, spec_ans = f['field'], f['spec']

        np.testing.assert_almost_equal(spc.field, field_ans)
        np.testing.assert_almost_equal(spc.spec, spec_ans)

    def test_from_array(self):
        field = np.linspace(3420, 3520, 256)
        abs_spec = cauchy(3450, 1).pdf(field) + cauchy(3470, 1).pdf(field) + cauchy(3490, 1).pdf(field)
        spec = np.gradient(abs_spec, field)

        spc = CWSpec(field, spec + 0.1, preprocess=True)

        np.testing.assert_almost_equal(spc.field, field)
        np.testing.assert_almost_equal(spc.spec, spec / np.trapz(abs_spec - min(abs_spec), field))

    def test_abs_first_moment(self):
        spc = CWSpec.from_file('test_data/Example_CW.DTA', preprocess=True)
        assert spc.abs_first_moment == 12.285957467546188

    def test_second_moment(self):
        spc = CWSpec.from_file('test_data/Example_CW.DTA', preprocess=True)
        assert spc.second_moment == 233.43611366615266


class TestUtils:

    def test_opt_phase(self):
        Vans = np.arange(100, dtype=complex) + np.random.rand(100)
        Vexp = Vans*np.exp(-1j * np.pi / 4)
        Vfit, phi_opt = utils.opt_phase(Vexp, return_params=True)
        np.testing.assert_almost_equal(Vfit, Vans)
        np.testing.assert_almost_equal(phi_opt, np.pi/4)

    def test_opt_phase_2d(self):
        Vans = np.tile(np.arange(100, dtype=complex), (9, 1)) + np.random.rand(9, 100)
        phi_ans = np.linspace(0, 2*np.pi, 10)[:-1]

        Vexp = Vans*np.exp(-1j * phi_ans)[:, None]
        Vfit, phi_fit = utils.opt_phase(Vexp, return_params=True)

        phi_fit -= 2*np.pi * np.floor(phi_fit / (2 * np.pi) )
        np.testing.assert_almost_equal(Vfit, Vans)
        np.testing.assert_almost_equal(phi_fit, phi_ans)

    @pytest.mark.parametrize('kwargs, expected', test_data)
    def test_generate_kernel(self, kwargs, expected):
        K, r, t = utils.generate_kernel(**kwargs)
        np.testing.assert_almost_equal(expected, K)

    def test_generate_kernel_nm(self):
        rnm = np.linspace(1.5, 10, 2**10)
        tnm = np.linspace(-1, 4.5, 2**10)

        Knm, _, _ = utils.generate_kernel(rnm, tnm)
        Kan, _, _ = utils.generate_kernel(r, t)

        np.testing.assert_almost_equal(Kan, Knm)

    @pytest.mark.parametrize('dta_file', DTAFiles)
    def test_fit_zero_time(self, dta_file):
        ex = DeerExp.from_file(dta_file, r=r)
        fit_time, fit_spec = utils.fit_zero_time(ex.raw_time, ex.raw_real)

        # plt.plot(fit_time, fit_spec.real)
        # plt.plot(fit_time, fit_spec.imag)
        # plt.axvline(0)
        # plt.axhline(1)
        # plt.show()
