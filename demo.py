"""
RHSIC / RKCIT demo for the paper's UI and CI data (Sections 5.1.1-5.1.2).
"""
import numpy as np
from threadpoolctl import threadpool_limits

from RHSIC import RHSIC_RFF_test
from RKCIT import RKCIT_test


funcs = {
    "gsign": lambda x: np.abs(np.random.normal(0, 1, size=x.shape)) * np.sign(x),
    "sd": lambda x: 20 * np.sin(4 * np.pi * (x ** 2)),
    "sine": lambda x: 1 + np.sin(3 * x) ** 2,
    "cube": lambda x: x ** 3,
    "cos": lambda x: np.cos(x),
}
func_names = ["gsign", "sd", "sine", "cube", "cos"]


def data_gen_1(n_samples, test_type, tau, noise="gaussian"):
    """DG I: X ~ U(-20, 20); H1: Y = sign * exp(-X**2) + noise.

    test_type=True selects H0: Y is independent noise. Here tau is the noise
    standard deviation sigma for Gaussian noise (the scale for Laplace noise).
    Both hypotheses return (x, y, s) with s=x.
    """
    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace
    else:
        raise ValueError("not implemented in this demo")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("DG I requires a positive noise scale tau (sigma).")

    x = np.random.uniform(-20, 20, size=(n_samples, 1))
    if test_type:
        y = sampler(0, tau, size=(n_samples, 1))
    else:
        signs = np.random.choice([-1, 1], (n_samples, 1), True)
        y = signs * np.exp(-x ** 2) + sampler(0, tau, size=(n_samples, 1))
    return x, y, x


def data_gen_2_index(n_samples, index, test_type, tau, noise="gaussian"):
    """DG II: shared noise creates dependence in the lower tau fraction of X.

    index=0,...,99 selects 25 function pairs, with four trials per pair.
    X=f1(ex)+eb in both hypotheses. Under H0, Y=f2(ey); under H1, eb is
    added to Y where X is below its sample tau-quantile. This tau is a 
    fraction, unlike the noise scale in DG I. Returns (x, y, x).
    """
    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace
    else:
        raise ValueError("not implemented in this demo")
    if not isinstance(index, (int, np.integer)) or not 0 <= index < 100:
        raise ValueError("DG II index must be an integer from 0 to 99.")
    if not np.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("DG II requires tau between 0 and 1.")

    keys = np.random.choice(range(5), 2)
    keys[0] = index // 4 // 5
    keys[1] = index // 4 % 5
    func1, func2 = (funcs[func_names[k]] for k in keys)
    x = sampler(size=(n_samples, 1))
    y = sampler(size=(n_samples, 1))
    x, y = func1(x), func2(y)
    eb = sampler(size=(n_samples, 1))
    x += eb
    if not test_type:
        threshold = np.percentile(x, tau * 100)
        y[x < threshold] += eb[x < threshold]
    return x, y, x


def data_gen_ci_index(n_samples, dim, index, test_type, tau, noise="gaussian"):
    """CI example: transform mean(Z)+noise, then add rare shared noise.
    X includes eb under both H0 and H1; only H1 adds eb to Y below the 
    sample tau-quantile of X. Returns (x,y,z,x).
    """
    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace
    else:
        raise ValueError("not implemented in this demo")
    if not isinstance(index, (int, np.integer)) or not 0 <= index < 100:
        raise ValueError("CI index must be an integer from 0 to 99.")
    if not isinstance(dim, (int, np.integer)) or dim < 1:
        raise ValueError("dim must be a positive integer.")
    if not np.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("CI requires tau between 0 and 1.")

    keys = np.random.choice(range(5), 2)
    keys[0] = index // 4 // 5
    keys[1] = index // 4 % 5
    func1, func2 = (funcs[func_names[k]] for k in keys)
    x = sampler(size=(n_samples, 1))
    y = sampler(size=(n_samples, 1))
    z = sampler(size=(n_samples, dim))
    mean_z = np.mean(z, axis=1).reshape(-1, 1)
    x += mean_z
    y += mean_z
    x, y = func1(x), func2(y)
    eb = sampler(size=(n_samples, 1))
    x += eb
    if not test_type:
        threshold = np.percentile(x, tau * 100)
        y[x < threshold] += eb[x < threshold]
    return x, y, z, x


def run_experiment(gen_type, num_samples, test_type, tau, noise="gaussian",
                   seed=0, num_trials=1, alpha=0.05, dim_z=10, ci_max_iter=100):
    """Run consecutive trial indices; return rows [seed, p_value, statistic]."""
    if gen_type not in ("dataset_1", "dataset_2", "conditional"):
        raise ValueError("Choose 'dataset_1' (DG I), 'dataset_2' (DG II), or 'conditional' (CI).")
    if not isinstance(num_samples, (int, np.integer)) or num_samples < 4:
        raise ValueError("num_samples must be an integer of at least 4.")
    if gen_type == "conditional" and num_samples % 2:
        raise ValueError("RKCIT requires an even num_samples.")
    if not isinstance(num_trials, (int, np.integer)) or num_trials < 1:
        raise ValueError("num_trials must be a positive integer.")
    if not isinstance(seed, (int, np.integer)) or seed < 0 or seed + num_trials > 100:
        raise ValueError("Use trial indices 0,...,99; for all 100 trials, set seed=0.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    name = {"dataset_1": "DG I", "dataset_2": "DG II", "conditional": "CI"}[gen_type]
    method = "RKCIT" if gen_type == "conditional" else "RHSIC"
    hypothesis = "H0" if test_type else "H1"
    parameter = "sigma" if gen_type == "dataset_1" else "tau"
    print(f"{name}: {hypothesis}, n={num_samples}, {parameter}={tau}, "
          f"noise={noise}, trials={num_trials}", flush=True)
    if gen_type == "conditional":
        print(f"dim_z={dim_z}, max_iter={ci_max_iter}", flush=True)
    results = []
    with threadpool_limits(limits=1):
        for trial_seed in range(seed, seed + num_trials):
            np.random.seed(trial_seed)
            if gen_type == "dataset_1":
                x, y, s = data_gen_1(num_samples, test_type, tau, noise)
            elif gen_type == "dataset_2":
                x, y, s = data_gen_2_index(num_samples, trial_seed, test_type, tau, noise)
            else:
                x, y, z, s = data_gen_ci_index(
                    num_samples, dim_z, trial_seed, test_type, tau, noise)

            # S=X is the reference variable in all three examples.
            if gen_type == "conditional":
                np.random.seed(trial_seed)
                p_value, stat = RKCIT_test(x, y, z, s, max_iter=ci_max_iter)
            else:
                # UI continues the generator's RNG stream without reseeding.
                p_value, stat = RHSIC_RFF_test(x, y, s)
            results.append((trial_seed, p_value, stat))
            print(f"seed={trial_seed}: {method} p-value={p_value:.6g}, "
                  f"statistic={stat:.6g}", flush=True)

    results = np.asarray(results)
    if num_trials > 1:
        finite = np.isfinite(results[:, 1:]).all(axis=1)
        print(f"Finite p-value/statistic: {finite.sum()}/{num_trials}")
        if finite.all():
            rate = np.mean(results[:, 1] < alpha)
            label = "Type I error" if test_type else "power"
            print(f"Raw returned-p {label} (p < {alpha}): {rate:.4f}")
        else:
            print("Rate omitted because some trials returned nonfinite values.")
    return results


if __name__ == "__main__":
    gen_type = "dataset_1"     # "dataset_1": DG I; "dataset_2": DG II; "conditional": CI
    num_samples = 1000
    test_type = False          # True: H0 / Type I error; False: H1 / power
    noise = "gaussian"
    seed = 0
    num_trials = 1             # Set 100, with seed=0, for the full trial set.
    alpha = 0.05
    tau = {"dataset_1": 0.5, "dataset_2": 0.01, "conditional": 0.03}[gen_type]
    dim_z = 10                # CI only; Figure 4 varies this from 1 to 10.
    ci_max_iter = 100          # CI only

    # Figure 5 UI settings: vary num_samples over
    # [500, 1000, 1500, 2000, 2500, 3000], keeping sigma=0.5 for DG I and
    # tau=0.01 for DG II; run both test_type=True and test_type=False.
    # Figure 4 DG II power: n=1000, test_type=False, and tau in
    # [0.030, 0.028, 0.026, 0.024, 0.022, 0.020, 0.018, 0.016, 0.014, 0.012, 0.010].
    # For DG I, change tau to the desired noise sigma.
    # CI: n=1000, tau=0.03, dim_z=1,...,10 for the dimension comparison;
    # use dim_z=10 and the same tau list above for the rare-fraction comparison.
    # Defaults: RHSIC 50 iterations / 2000 permutations; RKCIT 100 / 1000.
    results = run_experiment(gen_type, num_samples, test_type, tau, noise,
                             seed, num_trials, alpha, dim_z, ci_max_iter)

    # Optional: save every trial for subsequent plots or comparisons.
    # np.savetxt("results.csv", results, delimiter=",",
    #            header="seed,p_value,statistic", comments="")
