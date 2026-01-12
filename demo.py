from RHSIC import RHSIC_RFF_test
import numpy as np

def example_data(n_samples, test_type, tau, noise="gaussian"):
    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace

    func1 = lambda x: np.exp(-np.abs(x))
    func2 = lambda x: x**2
 
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
    # np.random.seed(0)
    num_samples = 500
    test_type = False
    tau = 0.1
    noise = "gaussian"

    # Generate data: s is the reference variable. If we don't know the ground truth reference variable, 
    # we can set s as either X or Y (for more details please refer to the paper).
    x, y, s = example_data(
            n_samples=num_samples, test_type=test_type, tau=tau, noise=noise
        )

    results = RHSIC_RFF_test(x, y, s)
    p_value, stat = results
    print(f"RHSIC test p-value: {p_value}, statistic: {stat}")
    