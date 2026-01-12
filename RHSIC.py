from tqdm import tqdm
import time
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from kerpy.GaussianKernel import GaussianKernel

#############################################
#          Some helper functions            #
#############################################

def more_than_tau_indices(array, tau):
    temp = np.where(np.array(array)>tau)[0]
    return temp

def kernel_matrix(data_x):
    kernelX = GaussianKernel()
    kernelX.set_width_empirical_hsic(data_x)

    data_x = stats.zscore(data_x, ddof=1, axis=0)
    data_x[np.isnan(data_x)] = 0.   

    Kx = kernelX.kernel(data_x)
    return Kx

def compute_rff(data_x, kernelX=GaussianKernel(), num_rfx=20):
    kernelX.rff_generate(num_rfx,dim=data_x.shape[1])
    phix = kernelX.rff_expand(data_x)
    return phix

def HSIC_rff_beta(phix, phiy, betas):
    m = phix.shape[0]
    if len(betas.shape) == 1:
        betas = betas.reshape(-1, 1)
    if np.sum(betas<0) > 0:
        betas[betas<0] += 1e-4
    phix_c = (phix - np.mean(phix * betas,axis=0)) * np.sqrt(betas)
    phiy_c = (phiy - np.mean(phiy * betas,axis=0)) * np.sqrt(betas)
    featCov = phix_c.T @ phiy_c 
    return np.linalg.norm(featCov)**2, np.linalg.norm(phix_c.T @ phix_c), np.linalg.norm(phiy_c.T @ phiy_c)


#############################################
#                RHSIC_RFF                  #
#############################################

def RHSIC_RFF_test(x, y, s, return_betas=False):
    '''RHSIC with RFF and alpha optimization.'''
    n = x.shape[0]
    ITERS = 50   #10000
    lam_reg1 = 1e-3 #1e-4
    lam_reg2 = 1e-3

    # NOTE: random split 
    x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=0.5, random_state=42)
    s = np.vstack([s_train, s_test])
    Ks = kernel_matrix(s)
    Ks_train = Ks[:s_train.shape[0], :s_train.shape[0]] + 1e-6 * np.eye(s_train.shape[0], dtype=np.float32)   # add small noise to diagonal to make it positive definite
    Ks_test = Ks[s_train.shape[0]:, :s_train.shape[0]]
    
    phix_test, phiy_test, phix_train, phiy_train = compute_rff(x_test), compute_rff(y_test), compute_rff(x_train), compute_rff(y_train)

    def objective(alpha):
        n = phix_train.shape[0]
        betas = Ks_train @ alpha
        HSIC, varx, vary = HSIC_rff_beta(phix_train, phiy_train, betas)
        reg1 = alpha @ Ks_train @ alpha
        reg2 = np.sum(betas * betas) / n
        return - np.log(HSIC) + np.log(varx) + np.log(vary) + lam_reg1 * reg1 + lam_reg2 * reg2

    def constraint_eq(alpha):
        n = Ks_train.shape[0]
        betas = Ks_train @ alpha
        return np.sum(betas) - n

    def constraint_ineq(alpha):
        betas = Ks_train @ alpha
        return betas

    constraints = [
        {'type': 'eq', 'fun': constraint_eq},
        {'type': 'ineq', 'fun': constraint_ineq}
    ]

    alpha0 = np.linalg.solve(Ks_train, np.ones(n//2, dtype=np.float32))
    print(f"alpha0's beta < 0: {np.sum(Ks_train @ alpha0 < 0)}")

    previous_x = [0]; count = [0]
    def callback_func(x):
        betas = Ks_train @ x
        print(f"iter: {count[-1]} | alpha diff: {np.sum(x-previous_x[-1]):.2f}, loss: {objective(x):.2f}")
        print(f"    beta < 0: {np.sum(betas<0):.2f}, min(betas): {np.min(betas):.2f}, sum(betas): {np.sum(betas):.2f}")
        previous_x[-1] = x; count[-1] += 1

    start = time.time()
    result = minimize(objective, alpha0, constraints=constraints, method='SLSQP', callback=callback_func, options={'disp': True, 'maxiter': ITERS})
    print(f"Time passed: {time.time()-start}s")

    optimized_train_betas = Ks_train @ result.x
    print(f"train beta > 1: {np.sum(optimized_train_betas > 1)}, sum: {np.sum(optimized_train_betas)}, <0: {np.sum(optimized_train_betas < 0)}")

    optimized_test_betas = Ks_test @ result.x
    print(f"test beta > 1: {np.sum(optimized_test_betas > 1)}, sum: {np.sum(optimized_test_betas)}, <0: {np.sum(optimized_test_betas < 0)}")

    index_train = more_than_tau_indices(optimized_train_betas, 1.0)
    index_test = more_than_tau_indices(optimized_test_betas, 1.0)
    drop_index_train = np.delete(np.arange(n//2), index_train)
    add_train = np.sum(optimized_train_betas[drop_index_train]) / len(index_train)
    optimized_train_betas[drop_index_train] = 0.0
    optimized_train_betas[index_train] = optimized_train_betas[index_train] + add_train
    print(f"\ntrain beta > 1: {np.sum(optimized_train_betas > 1)}, sum: {np.sum(optimized_train_betas)}, <0: {np.sum(optimized_train_betas < 0)}")

    drop_index_test = np.delete(np.arange(n//2), index_test)
    add_test = np.sum(optimized_test_betas[drop_index_test]) / len(index_test)
    optimized_test_betas[drop_index_test] = 0.0
    optimized_test_betas[index_test] = optimized_test_betas[index_test] + add_test
    print(f"test beta > 1: {np.sum(optimized_test_betas > 1)}, sum: {np.sum(optimized_test_betas)}, <0: {np.sum(optimized_test_betas < 0)}")
    ###############################################################################################################
    otrain_HSIC, otrain_varx, otrain_vary = HSIC_rff_beta(phix_train, phiy_train, np.ones(n//2, dtype=np.float32)) #len(index_train)
    print(f"original train HSIC: {otrain_HSIC}, obj: {otrain_HSIC/otrain_varx/otrain_vary}")

    train_HSIC, train_varx, train_vary = HSIC_rff_beta(phix_train, phiy_train, optimized_train_betas)
    print(f"reweighed train HSIC: {train_HSIC}, obj: {train_HSIC/train_varx/train_vary}")
    ###############################################################################################################
    
    otest_HSIC, otest_varx, otest_vary = HSIC_rff_beta(phix_test, phiy_test, np.ones(n//2, dtype=np.float32)) #len(index_test)
    print(f"original test HSIC: {otest_HSIC}, obj: {otest_HSIC/otest_varx/otest_vary}")

    test_HSIC, test_varx, test_vary = HSIC_rff_beta(phix_test, phiy_test, optimized_test_betas)
    print(f"reweighed test HSIC: {test_HSIC}, obj: {test_HSIC/test_varx/test_vary}")

    def compute_pvalue_perm(test_stat, phix, y, betas, n_perm=2000):
        stats = []
        temp_y = y.copy()
        for i in tqdm(range(n_perm), total=n_perm):
            perm_index = np.random.permutation(list(range(y.shape[0])))
            temp_y = y[perm_index]
            phiy_new = compute_rff(temp_y)
            temp_stat, _, _ = HSIC_rff_beta(phix, phiy_new, betas)
            stats.append(temp_stat)

        p = np.sum(np.array(stats) > test_stat) / n_perm
        return p

    p = compute_pvalue_perm(test_HSIC, phix_test, y_test, optimized_test_betas)

    if return_betas:
        return p, test_HSIC, optimized_test_betas, x_test, y_test
    
    return p, test_HSIC
    
