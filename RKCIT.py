from tqdm import tqdm
import torch
import torch.nn.utils as utils
from scipy.spatial.distance import cdist, pdist, squareform
from scipy import special, spatial
from sklearn.model_selection import train_test_split

import numpy as np
from scipy import stats
from KCI import KCI_CInd
from causallearn.utils.KCI.GaussianKernel import GaussianKernel


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def kernel_matrix(data_x):
    """
    Compute kernel matrix for data x and data y

    Parameters
    ----------
    data_x: input data for x (nxd1 array)
    data_y: input data for y (nxd2 array)

    Returns
    _________
    Kx: kernel matrix for data_x (nxn)
    Ky: kernel matrix for data_y (nxn)
    """
    kernelX = GaussianKernel()
    kernelX.set_width_empirical_hsic(data_x)

    data_x = stats.zscore(data_x, ddof=1, axis=0)
    data_x[np.isnan(data_x)] = 0.   # in case some dim of data_x is constant

    Kx = kernelX.kernel(data_x)
    return Kx


def objective(epsilons, Kx, Ky, Kz, Ks, epsilon, lam_reg1, lam_reg2):
    n = Kx.shape[0]
    betas = torch.exp(epsilons) / torch.sum(torch.exp(epsilons)) * n
    alpha = torch.linalg.solve(Ks, betas)
    
    KCI, VarxHS, VaryHS = KCIb(Kx, Ky, Kz, epsilon, betas)

    reg1 = alpha @ Ks @ alpha
    reg2 = alpha @ Ks @ Ks @ alpha / n - 2 * torch.ones(n, dtype=torch.float64).to(device) @ Ks @ alpha / n + 1

    return - torch.log(KCI) + torch.log(VarxHS) + torch.log(VaryHS) + lam_reg1 * reg1 + lam_reg2 * reg2

def KCIb(Kx, Ky, Kz, epsilon, betas=None):
    n = Kx.shape[0]
    Hb = torch.diag(betas) @ (torch.eye(n, dtype=torch.float64, device=device) - torch.ones((n, n), dtype=torch.float64, device=device) @ torch.diag(betas) / n) 
    Rb = epsilon * torch.linalg.pinv(Kz @ Hb + epsilon * torch.eye(n, dtype=torch.float64, device=device))
    KCI, VarxHS, VaryHS = torch.trace(Rb @ (Kx) @ Rb.T @ Hb @ Rb @ (Ky) @ Rb.T @ Hb), torch.trace(Rb @ (Kx) @ Rb.T @ Hb), torch.trace(Rb @ (Ky) @ Rb.T @ Hb)
    return KCI, VarxHS, VaryHS


def get_Gkernel_matrix(X: np.ndarray, Y=None, width=None):
    """
    Computes the Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)=exp(-0.5* ||x-y||**2 *self.width)
    """
    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert (np.shape(X)[1] == np.shape(Y)[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')
    
    if width is None:
        n = X.shape[0]
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        theta = 1.0 / (width ** 2)
        width = theta / X.shape[1]

    K = np.exp(-0.5 * sq_dists * width)
    return K

def get_restricted_permutation(T, shuffle_neighbors, neighbors, order):

    restricted_permutation = np.zeros(T, dtype=np.int32)
    used = np.array([], dtype=np.int32)

    for sample_index in order:
        m = 0
        use = neighbors[sample_index, m]

        while ((use in used) and (m < shuffle_neighbors - 1)):
            m += 1
            use = neighbors[sample_index, m]

        restricted_permutation[sample_index] = use
        used = np.append(used, use)

    return restricted_permutation


def RKCIT_test(x, y, z, s, approx='perm_knn'):
    ITERS = 1000
    epsilon = 1e-3
    LR = 1e-2
    lam_reg1 = 1e-4     # 0.01 1e-8
    lam_reg2 = 0.1        # 1

    n = x.shape[0]
    x_train, x_test, y_train, y_test, z_train, z_test, s_train, s_test = train_test_split(x, y, z, s, test_size=0.5, random_state=42)
    s = np.vstack([s_train, s_test])
    Ks = torch.from_numpy(get_Gkernel_matrix(X=s)).type(torch.float64).to(device)
    Ks_train = Ks[:n//2, :n//2] + 1e-6 * torch.eye(n//2, dtype=torch.float64, device=device)   # add small noise to diagonal to make it positive definite
    Ks_test = Ks[n//2:, :n//2]  
    calculator = KCI_CInd()
    Kx_test, Ky_test, Kzx_test, _, _, _, _ = calculator.kernel_matrix(x_test, y_test, z_test)
    Kx_train, Ky_train, Kzx_train, _, _, _, _ = calculator.kernel_matrix(x_train, y_train, z_train)
    Kx_test, Ky_test, Kzx_test = torch.from_numpy(Kx_test).to(device), torch.from_numpy(Ky_test).to(device), torch.from_numpy(Kzx_test).to(device)
    Kx_train, Ky_train, Kzx_train = torch.from_numpy(Kx_train).to(device), torch.from_numpy(Ky_train).to(device), torch.from_numpy(Kzx_train).to(device)
    params = torch.ones(n//2, dtype=torch.float64, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([params], lr=LR)

    for step in tqdm(range(ITERS), total=ITERS):
        loss = objective(params, Kx_train, Ky_train, Kzx_train, Ks_train, epsilon, lam_reg1, lam_reg2)
        
        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_([params], max_norm=1.0)
        optimizer.step()
        
    optimized_train_betas = torch.exp(params - torch.max(params)) / torch.sum(torch.exp(params - torch.max(params))) * (n//2)
    print(f"train beta > 0.4: {torch.sum(optimized_train_betas > 0.4)}, sum: {torch.sum(optimized_train_betas)}")

    alpha = torch.linalg.solve(Ks_train, optimized_train_betas)
    optimized_test_betas = Ks_test @ alpha
    print(f"test beta > 0.4: {torch.sum(optimized_test_betas > 0.4)}, sum: {torch.sum(optimized_test_betas)}")

    otest_KCI, otest_varx, otest_vary = KCIb(Kx_test, Ky_test, Kzx_test, epsilon, torch.ones(n//2, dtype=torch.float64).to(device))
    print(f"original test KCI: {otest_KCI}, obj: {otest_KCI/otest_varx/otest_vary}")

    test_KCI, test_varx, test_vary = KCIb(Kx_test, Ky_test, Kzx_test, epsilon, optimized_test_betas)
    print(f"reweighed test KCI: {test_KCI}, obj: {test_KCI/test_varx/test_vary}")
    
    def compute_pvalue_original(test_stat, data_x=None, data_y=None, data_z=None):
        """
        Main function: compute the p value and return it together with the test statistic
        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)
        data_z: input data for z (nxd3 array)

        Returns
        _________
        pvalue: p value
        test_stat: test statistic
        """
        calc = KCI_CInd(kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian')
        # print(np.shape(data_x), np.shape(data_y), np.shape(data_z))
        Kx, Ky, Kzx, Kzy, kernelX, kernelY, kernelZ = calc.kernel_matrix(data_x, data_y, data_z)
        _, KxR, KyR = calc.KCI_V_statistic(Kx, Ky, Kzx, Kzy)
        uu_prod, size_u = calc.get_uuprod(KxR, KyR)
        from scipy import stats
        null_samples = calc.null_sample_spectral(uu_prod, size_u, Kx.shape[0])
        pvalue1 = sum(null_samples > _) / float(calc.nullss)
        p = sum(null_samples > test_stat.detach().numpy()) / float(calc.nullss)
        return pvalue1, p, test_stat

    def compute_pvalue_perm_o(test_stat, data_x=None, data_y=None, data_z=None, betas=None, epsilon=1e-3, n_perm=1000):
        n = data_x.shape[0]
        stats = []
        for i in tqdm(range(n_perm), total=n_perm):
            perm_index = np.random.permutation(list(range(n)))
            temp_y = data_y[perm_index]
            temp_Ky = kernel_matrix(temp_y)
            temp_stat, _, _ = KCIb(Kx_test, temp_Ky, Kzx_test, epsilon, betas)
            stats.append(temp_stat.detach().cpu().numpy())

        p = np.sum(np.array(stats) > test_stat) / n_perm
        return p, test_stat
    
    def compute_pvalue_perm_knn(test_stat, data_x=None, data_y=None, data_z=None, betas=None, epsilon=1e-3, n_perm=1000):
        array = np.vstack((data_x.T, data_y.T, data_z.T))
        # xyz is the dimension indicator
        xyz = np.array([0 for i in range(data_x.shape[1])] +
                        [1 for i in range(data_y.shape[1])] +
                        [2 for i in range(data_z.shape[1])])
        value = test_KCI.detach().cpu().numpy()

        seed = 42
        shuffle_neighbors = 5
        sig_samples = 500
        random_state = np.random.default_rng(seed)
        
        dim, T = array.shape

        # max_neighbors = max(1, int(max_neighbor_ratio*T))
        x_indices = np.where(xyz == 0)[0]
        y_indices = np.where(xyz == 1)[0]
        z_indices = np.where(xyz == 2)[0]

        # # Get nearest neighbors around each sample point in Z
        # z_array = np.fastCopyAndTranspose(array[z_indices, :])
        z_array = array[z_indices, :].T.copy()
        tree_xyz = spatial.cKDTree(z_array)
        neighbors = tree_xyz.query(z_array,
                                    k=shuffle_neighbors,
                                    p=np.inf,
                                    eps=0.)[1].astype(np.int32)



        null_dist = np.zeros(sig_samples)
        for sam in tqdm(range(sig_samples), total=sig_samples):

            # Generate random order in which to go through indices loop in
            # next step
            order = random_state.permutation(T).astype(np.int32)

            # Shuffle neighbor indices for each sample index
            for i in range(len(neighbors)):
                random_state.shuffle(neighbors[i])
            # neighbors = self.random_state.permuted(neighbors, axis=1)
            
            # Select a series of neighbor indices that contains as few as
            # possible duplicates
            restricted_permutation = get_restricted_permutation(
                    T=T,
                    shuffle_neighbors=shuffle_neighbors,
                    neighbors=neighbors,
                    order=order)

            array_shuffled = np.copy(array)
            for i in y_indices:
                array_shuffled[i] = array[i, restricted_permutation]

            y_new = array_shuffled[xyz==1].T

            temp_Ky = torch.from_numpy(kernel_matrix(y_new)).to(device)
            temp_stat, _, _ = KCIb(Kx_test, temp_Ky, Kzx_test, epsilon, betas)
            null_dist[sam] = temp_stat.detach().cpu().numpy()

        pval = (null_dist >= value).mean()

        return pval
    
    
    if approx == "perm_o":
        p, Stat = compute_pvalue_perm_o(test_KCI.detach().cpu().numpy(), data_x=x_test, data_y=y_test, data_z=z_test, 
                                      betas=optimized_test_betas, epsilon=epsilon)
    elif approx == "perm_knn":
        p = compute_pvalue_perm_knn(test_KCI.detach().cpu().numpy(), data_x=x_test, data_y=y_test, data_z=z_test, 
                                      betas=optimized_test_betas, epsilon=epsilon)
    elif approx == 'origin':
        pvalue1, p, _ = compute_pvalue_original(test_KCI, x_test, y_test, z_test)
    return p, test_KCI.detach().cpu().numpy()
