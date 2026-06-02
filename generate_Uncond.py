import numpy as np

funcs = {
    "gsign": lambda x: np.abs(np.random.normal(0,1,size=x.shape)) * np.sign(x) ,
    "sd": lambda x: 20 * np.sin(4 * np.pi * (x ** 2)),
    "sine": lambda x: (1 + np.sin(3 * x) ** 2),
    "cube": lambda x: x**3,
    "cos": lambda x: np.cos(x),
}
func_names = ["gsign", "sd", "sine", "cube", "cos"]


def data_gen_1(n_samples, test_type, tau, noise="gaussian"):
    """Generate DG I.

    If `test_type=True`, `Y` is pure noise for Type I error evaluation. If
    `test_type=False`, `Y` depends nonlinearly on `X` and `tau` is the noise
    scale for the power setting. Returns `(x, y, s)` with `s=x`.
    """
    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace

    x = np.random.uniform(-20, 20, size=(n_samples, 1))
    if test_type:
        y = sampler(0, tau, size=(n_samples, 1))
        return x, y, x
    else:
        signs_ = np.random.choice([-1, 1], (n_samples, 1), True)
        y = signs_ * np.exp(-x**2) + sampler(0, tau, size=(n_samples, 1))
        return x, y, x
    

def data_gen_2_index(n_samples, index, test_type, tau, noise="gaussian"):
    """Generate DG II with one indexed pair of nonlinear functions.

    If `test_type=True`, the transformed `X` and `Y` stay independent for Type
    I error evaluation. If `test_type=False`, the shared noise `eb` is added to
    `Y` only when `X` is below the `tau`-quantile, creating thresholded
    dependence. Returns `(x, y, s)` with `s=x`.
    """
    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace
    keys = np.random.choice(range(5), 2)
    keys[0] = index // 4 // 5
    keys[1] = index // 4 % 5
    pnl_funcs = [func_names[k] for k in keys]

    func1 = funcs[pnl_funcs[0]]
    func2 = funcs[pnl_funcs[1]]
 
    x = sampler(size=(n_samples, 1))
    y = sampler(size=(n_samples, 1))
    x, y = func1(x), func2(y)
    eb = sampler(size=(n_samples, 1))
    x += eb

    if test_type:
        return x, y, x
    else:
        tresh = np.percentile(x, tau*100)
        y[x < tresh] += eb[x < tresh]
        return x, y, x
    

def data_gen_3_index(n_samples, index, test_type, tau, noise="gaussian"):
    """Generate DG III with an auxiliary threshold variable.

    This is DG II with a separate uniform variable `s` controlling the regime.
    If `test_type=True`, the variables stay independent for Type I error
    evaluation. If `test_type=False`, `eb` is added to `Y` only when `s < tau`.
    Returns `(x, y, s)`.
    """
    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace
    keys = np.random.choice(range(5), 2)
    keys[0] = index // 4 // 5
    keys[1] = index // 4 % 5
    pnl_funcs = [func_names[k] for k in keys]

    func1 = funcs[pnl_funcs[0]]
    func2 = funcs[pnl_funcs[1]]
 
    x = sampler(size=(n_samples, 1))
    y = sampler(size=(n_samples, 1))
    s = np.random.uniform(0, 1, size=(n_samples, 1))
    x, y = func1(x), func2(y)
    eb = sampler(size=(n_samples, 1))
    x += eb

    if test_type:
        return x, y, s
    else:
        y[s < tau] += eb[s < tau]
        return x, y, s
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # x, y, s = data_gen_3_index(1000, 1, False, 0.1)
    x, y, s = data_gen_2_index(1000, 1, False, 0.1)
    plt.scatter(x, y, c=s)
    plt.show()