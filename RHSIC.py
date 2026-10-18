"""
Copyright (c) 2016 oxmlcs
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 *
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 *
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the author.
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


def _compute_rff(data_x):
    rff_num = 20
    unit_rff_freq = np.random.randn(int(rff_num / 2), data_x.shape[1])
    rff_freq = unit_rff_freq * np.sqrt(1.0 / 1.0)
    xdotw = np.dot(data_x, rff_freq.T)
    return np.sqrt(2. / rff_num) * np.concatenate((np.cos(xdotw), np.sin(xdotw)), axis=1)


def _hsic(phix, phiy, betas, _observed_arrays=None):
    if len(betas.shape) == 1:
        betas = betas.reshape(-1, 1)
    if np.sum(betas < 0) > 0:
        betas[betas < 0] += 1e-4
    phix_c = (phix - np.mean(phix * betas, axis=0)) * np.sqrt(betas)
    phiy_c = (phiy - np.mean(phiy * betas, axis=0)) * np.sqrt(betas)
    featCov = phix_c.T @ phiy_c
    if _observed_arrays is not None:
        _observed_arrays.extend((phix_c, phiy_c, featCov))
    return np.linalg.norm(featCov)**2, np.linalg.norm(phix_c.T @ phix_c), np.linalg.norm(phiy_c.T @ phiy_c)


def RHSIC_RFF_test(x, y, s, return_betas=False, *, max_iter=50, n_perm=2000, verbose=False):
    x, y, s = _validate_inputs(x, y, s)
    _validate_options(max_iter, n_perm)
    n = x.shape[0]
    x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
        x, y, s, test_size=0.5, random_state=42)
    Ks = _kernel_matrix(np.vstack([s_train, s_test]))
    Ks_train = Ks[:s_train.shape[0], :s_train.shape[0]] + 1e-6 * np.eye(s_train.shape[0], dtype=np.float32)
    Ks_test = Ks[s_train.shape[0]:, :s_train.shape[0]]
    phix_test, phiy_test, phix_train, phiy_train = (_compute_rff(x_test), _compute_rff(y_test),
                                                 _compute_rff(x_train), _compute_rff(y_train))

    def objective(alpha):
        n = phix_train.shape[0]
        betas = Ks_train @ alpha
        HSIC, varx, vary = _hsic(phix_train, phiy_train, betas)
        reg1 = alpha @ Ks_train @ alpha
        reg2 = np.sum(betas * betas) / n
        return - np.log(HSIC) + np.log(varx) + np.log(vary) + 1e-3 * reg1 + 1e-3 * reg2

    def constraint_eq(alpha):
        return np.sum(Ks_train @ alpha) - Ks_train.shape[0]

    def constraint_ineq(alpha):
        return Ks_train @ alpha

    constraints = [{'type': 'eq', 'fun': constraint_eq}, {'type': 'ineq', 'fun': constraint_ineq}]
    alpha0 = np.linalg.solve(Ks_train, np.ones(n // 2, dtype=np.float32))

    def callback(alpha):
        loss = objective(alpha)
        if verbose:
            print('objective:', loss)

    result = minimize(objective, alpha0, constraints=constraints, method='SLSQP', callback=callback,
                      options={'disp': verbose, 'maxiter': max_iter})
    train_betas = Ks_train @ result.x
    test_betas = Ks_test @ result.x
    for betas, size in ((train_betas, n // 2), (test_betas, n - n // 2)):
        index = np.where(np.array(betas) > 1.0)[0]
        drop_index = np.delete(np.arange(size), index)
        add = np.sum(betas[drop_index]) / len(index)
        betas[drop_index] = 0.0
        betas[index] = betas[index] + add
    observed = {}
    observed_arrays = []
    observed['original_train'] = _hsic(phix_train, phiy_train, np.ones(n // 2, dtype=np.float32), _observed_arrays=observed_arrays)
    observed['weighted_train'] = _hsic(phix_train, phiy_train, train_betas, _observed_arrays=observed_arrays)
    observed['original_test'] = _hsic(phix_test, phiy_test, np.ones(phix_test.shape[0], dtype=np.float32), _observed_arrays=observed_arrays)
    observed['weighted_test'] = _hsic(phix_test, phiy_test, test_betas, _observed_arrays=observed_arrays)
    test_stat = observed['weighted_test'][0]
    null = []
    for _ in range(n_perm):
        perm_index = np.random.permutation(list(range(y_test.shape[0])))
        phiy_new = _compute_rff(y_test[perm_index])
        temp_stat, _, _ = _hsic(phix_test, phiy_new, test_betas)
        null.append(temp_stat)
    null = np.asarray(null)
    p = np.sum(null > test_stat) / n_perm
    values = (p, test_stat, test_betas, x_test, y_test) if return_betas else (p, test_stat)
    return values
