"""
MIT License

Copyright (c) 2022 cmu-phil

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

def _validate_inputs(*arrays):
    result = []
    for value in arrays:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[1] == 0 or not np.all(np.isfinite(array)):
            raise ValueError("Inputs must be finite 2D arrays with at least one column.")
        result.append(array)
    n = result[0].shape[0]
    if n < 4 or any(array.shape[0] != n for array in result):
        raise ValueError("Inputs must have equal row counts and at least four rows.")
    return result


def _validate_options(max_iter, n_perm):
    for name, value in (("max_iter", max_iter), ("n_perm", n_perm)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(name + " must be a positive integer.")


def _kernel_matrix(data_x):
    # Inline empirical HSIC width and Gaussian kernel from the frozen dependency.
    n = data_x.shape[0]
    width = 0.8 if n < 200 else (0.5 if n < 1200 else 0.3)
    theta = 1.0 / (width ** 2)
    width = theta * data_x.shape[1]
    data_x = stats.zscore(data_x, ddof=1, axis=0)
    data_x[np.isnan(data_x)] = 0.
    sq_dists = squareform(pdist(data_x, 'sqeuclidean'))
    return np.exp(-0.5 * sq_dists * width)


def random_fourier_features_np(x, w=None, b=None, num_f=None, sigma=None, seed=None):
    if num_f is None:
        num_f = 25
    if seed is not None:
        np.random.seed(seed)
    r, c = x.shape
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        if sigma == 0:
            sigma = 1
        w = 1.0 / sigma * np.random.randn(num_f, c)
        b = np.tile(2 * np.pi * np.random.rand(num_f), (r, 1)).T
    feat = np.sqrt(2) * np.cos(np.matmul(w, x.T) + b)
    return {'feat': feat.T, 'w': w, 'b': b}

def normalize_np(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

def cov_np(x, y=None):
    if y is None:
        return np.cov(x.T)
    data = np.concatenate((x, y), axis=1)
    return np.cov(data.T)[:x.shape[1], x.shape[1]:]

def RCI_pre_np(x, y, z, num_f=100, num_f2=5, seed=None):
    z = z[:, np.std(z, axis=0) > 0]
    if z.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return 0
    r = x.shape[0]
    r1 = min(r, 500)
    x = normalize_np(x)
    y = normalize_np(y)
    z = normalize_np(z)
    x = np.hstack([x, z])
    dist_z = pdist(z[:r1], metric='euclidean')
    sigma_z = np.median(squareform(dist_z))
    four_z = random_fourier_features_np(z, num_f=num_f, sigma=sigma_z, seed=seed)
    dist_x = pdist(x[:r1], metric='euclidean')
    sigma_x = np.median(squareform(dist_x))
    four_x = random_fourier_features_np(x, num_f=num_f2, sigma=sigma_x, seed=seed)
    dist_y = pdist(y[:r1], metric='euclidean')
    sigma_y = np.median(squareform(dist_y))
    four_y = random_fourier_features_np(y, num_f=num_f2, sigma=sigma_y, seed=seed)
    f_x = normalize_np(four_x['feat'])
    f_y = normalize_np(four_y['feat'])
    f_z = normalize_np(four_z['feat'])
    return (f_x, f_y, f_z)

def RCI_reg_np(f_x, f_y, f_z, epsilon=0.001):
    r = f_x.shape[0]
    num_f_z = f_z.shape[1]

    def calc_pHp(p1, p2):
        term1 = cov_np(p1, p2)
        return term1
    Czz = calc_pHp(f_z, f_z)
    i_Czz = np.linalg.pinv(Czz + np.eye(num_f_z) * epsilon)
    Cxz = calc_pHp(f_x, f_z)
    Czy = calc_pHp(f_z, f_y)
    z_i_Czz = np.matmul(f_z, i_Czz)
    e_x_z = np.matmul(z_i_Czz, Cxz.T)
    e_y_z = np.matmul(z_i_Czz, Czy)
    res_x = f_x - e_x_z
    res_y = f_y - e_y_z
    return (res_x, res_y)

def HSIC_rff_beta_np(phix, phiy, betas, _observed_arrays=None):
    if len(betas.shape) == 1:
        betas = betas.reshape(-1, 1)
    if np.sum(betas < 0) > 0:
        betas[betas < 0] = 0.0
    phix_c = (phix - np.mean(phix * betas, axis=0)) * np.sqrt(betas)
    phiy_c = (phiy - np.mean(phiy * betas, axis=0)) * np.sqrt(betas)
    featCov = phix_c.T @ phiy_c
    if _observed_arrays is not None:
        _observed_arrays.extend((phix_c, phiy_c, featCov))
    return (np.linalg.norm(featCov) ** 2, np.linalg.norm(phix_c.T @ phix_c), np.linalg.norm(phiy_c.T @ phiy_c))


def RKCIT_test(x, y, z, s, *, max_iter=100, n_perm=1000, verbose=False):
    x, y, z, s = _validate_inputs(x, y, z, s)
    _validate_options(max_iter, n_perm)
    n = x.shape[0]
    if n % 2:
        raise ValueError("RKCIT requires an even row count for equal train/test halves.")
    x_train, x_test, y_train, y_test, z_train, z_test, s_train, s_test = train_test_split(
        x, y, z, s, test_size=0.5, random_state=42)
    for xi, yi, zi in ((x_train, y_train, z_train), (x_test, y_test, z_test)):
        if not np.any(np.std(zi, axis=0) > 0) or np.std(xi) == 0 or np.std(yi) == 0:
            raise ValueError("Each split needs nonconstant x/y and at least one nonconstant z column.")
    Ks = _kernel_matrix(np.vstack([s_train, s_test]))
    Ks_train = Ks[:n // 2, :n // 2] + 1e-6 * np.eye(n // 2)
    Ks_test = Ks[n // 2:, :n // 2]
    f_x_train, f_y_train, f_z_train = RCI_pre_np(x_train, y_train, z_train, num_f=100, num_f2=100)
    f_x_test, f_y_test, f_z_test = RCI_pre_np(x_test, y_test, z_test, num_f=100, num_f2=100)
    res_x_train, res_y_train = RCI_reg_np(f_x_train, f_y_train, f_z_train, 1e-3)

    def objective(alpha):
        betas = Ks_train @ alpha
        KCI, VarxHS, VaryHS = HSIC_rff_beta_np(res_x_train, res_y_train, betas)
        dependence = - np.log(KCI) + np.log(VarxHS) + np.log(VaryHS)
        reg1 = alpha @ Ks_train @ alpha
        reg2 = np.sum(betas * betas) / (n // 2)
        return dependence + 1e-6 * reg1 + 0.1 * reg2

    def constraint_ineq(alpha):
        return Ks_train @ alpha

    def constraint_eq(alpha):
        return np.abs(np.sum(Ks_train @ alpha) - n // 2)

    constraints = [{'type': 'ineq', 'fun': constraint_ineq}, {'type': 'eq', 'fun': constraint_eq}]
    alpha0 = np.linalg.solve(Ks_train, np.ones(n // 2))
    initial_objective = objective(alpha0)
    if verbose:
        print('initial objective:', initial_objective)

    def callback(alpha):
        loss = objective(alpha)
        if verbose:
            print('objective:', loss)

    result = minimize(objective, alpha0, constraints=constraints, method='SLSQP', callback=callback,
                      options={'disp': verbose, 'maxiter': max_iter})
    train_betas = Ks_train @ result.x
    test_betas = Ks_test @ result.x
    for betas in (train_betas, test_betas):
        index = np.where(np.array(betas) > 1.0)[0]
        drop_index = np.delete(np.arange(n // 2), index)
        add = np.sum(betas[drop_index]) / len(index)
        betas[drop_index] = 0.0
        betas[index] = betas[index] + add
    res_x_test, res_y_test = RCI_reg_np(f_x_test, f_y_test, f_z_test, 1e-3)
    observed = {}
    observed_arrays = []
    observed['original_train'] = HSIC_rff_beta_np(res_x_train, res_y_train, np.ones(n // 2), _observed_arrays=observed_arrays)
    observed['weighted_train'] = HSIC_rff_beta_np(res_x_train, res_y_train, train_betas, _observed_arrays=observed_arrays)
    observed['original_test'] = HSIC_rff_beta_np(res_x_test, res_y_test, np.ones(n // 2), _observed_arrays=observed_arrays)
    observed['weighted_test'] = HSIC_rff_beta_np(res_x_test, res_y_test, test_betas, _observed_arrays=observed_arrays)
    test_stat = observed['weighted_test'][0]
    null = []
    for _ in range(n_perm):
        perm_index = np.random.permutation(list(range(res_y_test.shape[0])))
        temp_stat, _, _ = HSIC_rff_beta_np(res_x_test, res_y_test[perm_index], test_betas)
        null.append(temp_stat)
    null = np.asarray(null)
    p = np.sum(null > test_stat) / n_perm
    values = (p, test_stat)
    return values
