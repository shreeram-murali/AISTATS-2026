import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import shutil
import dill

from typing import Callable
from torchvision import datasets
from scipy import optimize
from time import perf_counter
from reml import NadarayaWatsonClassifier
from scipy.special import gamma, kv

rng = np.random.default_rng()

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
plt.style.use('seaborn-v0_8-paper')

"""
Hyperparameter Optimisation Functions: Log Marginal Likelihood
"""

def kernel_matrix(X: np.ndarray, bandwidth: float, kernel: Callable, args: dict):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    indices = np.triu_indices(n_samples)
    distances = np.linalg.norm(X[indices[0]] - X[indices[1]], axis=1)
    K[indices] = kernel(distances / bandwidth, **args)
    K[(indices[1], indices[0])] = K[indices]

    return K 

def lml(X, y, bandwidth, kernel, kernel_kwargs, noise):
    n_samples = X.shape[0]
    K = kernel_matrix(X, bandwidth, kernel, kernel_kwargs)
    # check kernel assumptions
    # rbf kernel might be returning too small values 
    K_noisy = K + noise**2 * np.eye(n_samples)

    try:
        L = np.linalg.cholesky(K_noisy)
        log_marginal_likelihood = 0
        n_classes = y.shape[1]

        for c in range(n_classes):
            y_c = y[:, c]
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_c))
            lml_c = -0.5 * np.dot(y_c, alpha) - np.sum(np.log(np.diag(L))) - n_samples / 2 * np.log(2 * np.pi)
            # lml_c = -0.5 * np.dot(y_c, alpha) - n_samples / 2 * np.log(2 * np.pi)
            log_marginal_likelihood += lml_c
        
        # print("[LOG] Cholesky decomposition worked", flush=True)

    except np.linalg.LinAlgError:
        print("[WARNING] Cholesky decomposition didn't work", flush=True)
        return -1e10

    return log_marginal_likelihood

def optimize_theta(X, y, kernel_func, initial_params, param_bounds):
    iteration_count = [0]
    
    def objective(params):
        s, bandwidth, noise = params
        lml_value = lml(X, y, bandwidth, kernel_func, {'s': s}, noise)
        
        iteration_count[0] += 1
        print(f"[OPTIMIZATION] Iteration {iteration_count[0]}: s={s:.2e}, bandwidth={bandwidth:.2f}, noise={noise:.2f}, LML={lml_value:.2f}", flush=True)
        
        return -lml_value

    result = optimize.minimize(objective, initial_params, method='L-BFGS-B', bounds=param_bounds, options={'maxiter': 200})

    return result.x, -result.fun

def save_optimized_classifier():
    ... 

"""
Defining our kernel functions
"""

def K_rbf(x, s):
    sqe = np.minimum(s ** 2 * np.exp(-(x**2/2)), 1.0)
    if isinstance(sqe, np.ndarray):
        sqe[x > 1] = 1e-10
    else:
        sqe = 1e-10 if x > 1 else sqe
    return sqe


def K_rbf_standard(x, s):
    return np.minimum(s**2 * np.exp(-x**2 / 2), 1.)

# def K_rbf(x, s):
#     se = s ** 2 * np.exp(-(x**2/2)) 
#     if se >= 1.:
#         se = 1.
#     return se 

def K_boxcar(x):
    return 0.5*x if x <= 1. else 1e-10


def K_matern(x, s, v, l):
    if np.isscalar(x) and x == 0:
        x = 1e-10
    elif isinstance(x, np.ndarray):
        x = x.copy()
        x[x==0] = 1e-10
    
    part1 = 2 ** (1 - v) / gamma(v)
    part2 = (np.sqrt(2 * v) * np.abs(x) / l) ** v
    part3 = kv(v, np.sqrt(2 * v) * np.abs(x) / l)
    
    return s**2 * part1 * part2 * part3


"""
Utility Functions
"""

def load_train_set(n_train, train_set):
    num_train = n_train

    train_set_array = train_set.data.numpy()
    x_train = train_set_array.reshape(train_set_array.shape[0], -1)[:num_train]
    x_train = x_train / 255.0  # Normalize to [0, 1]

    y_train = train_set.targets.numpy()[:num_train]

    y_train_one_hot = np.zeros((num_train, 10))
    for i in range(num_train):
        y_train_one_hot[i, y_train[i]] = 1

    res = {'x_train': x_train, 'y_train': y_train, 'y_train_one_hot': y_train_one_hot}
    return res

def load_test_set(n_test, test_set):
    test_set_array = test_set.data.numpy()
    y_test_all = test_set.targets.numpy()
    test_indices = []

    for digit in range(10):
        digit_indices = np.where(y_test_all == digit)[0]
        if len(digit_indices) > 0:
            # Select one random sample of this digit
            selected_idx = rng.choice(digit_indices, 1)[0]
            test_indices.append(selected_idx)

    remaining = n_test - len(test_indices)
    if remaining > 0:
        remaining_indices = np.setdiff1d(np.arange(len(y_test_all)), test_indices)
        additional_indices = rng.choice(remaining_indices, remaining, replace=False)
        test_indices.extend(additional_indices)

    test_indices = np.array(test_indices[:n_test])
    x_test = test_set_array[test_indices].reshape(n_test, -1)
    x_test = x_test / 255.0
    y_test = y_test_all[test_indices]

    res = {'x_test': x_test, 'y_test': y_test}
    return res

"""
Main function
"""
def main():
    NUM_TRAIN = 30000
    NUM_TEST = 500
    SUBSET_SIZE = 1000 # we pick a subset because it would make this faster 
    # SIGMA_N = 0.5

    train_set = datasets.MNIST('./data', train=True, download=True)
    test_set = datasets.MNIST('./data', train=False, download=True)

    print("[LOG] Loading MNIST data", flush=True)
    dataset = load_train_set(NUM_TRAIN, train_set)
    x_train, y_train, y_train_onehot = dataset.values()
    validation = load_test_set(NUM_TEST, test_set)
    x_test, y_test = validation.values()
    # print("[LOG] Loaded training and validation sets", flush=True)

    print("[LOG] Computing a baseline with initial parameters", flush=True) 
    
    # NOTE: here's where you set initial hyperparameter values 
    initial_bandwidth = 7.5
    initial_s = 1.
    initial_sigma_n = 4.
    # --- --- ---
    
    sigma = 0.25
    lipschitz = 0.02

    nwc_initial = NadarayaWatsonClassifier(
        bandwidth=initial_bandwidth, 
        lipschitz_constant=lipschitz, 
        noise_parameter=sigma, 
        ck=1., 
        kernel=K_rbf, 
        kernel_kwargs={'s': initial_s}
    )

    baseline_predictions = []
    baseline_probabilities = []
    baseline_certainties = []

    for i in range(NUM_TEST):
        probs, bounds = nwc_initial.predict(x_test[i], x_train, y_train_onehot)
        bounds = min(bounds, 1.)
        predicted_class = np.argmax(probs)
        baseline_predictions.append(predicted_class)
        baseline_probabilities.append(probs)
        baseline_certainties.append(bounds)

    
    print("[LOG] Computing the baseline statistics", flush=True)  
    baseline_accuracy = sum(baseline_predictions[i] == y_test[i] for i in range(NUM_TEST)) / NUM_TEST * 100
    # baseline_mean_bound = np.mean([np.mean(bounds) for bounds in baseline_certainties])
    baseline_mean_bound = np.mean(np.array(baseline_certainties))
    print(f"[METRIC]\tBaseline accuracy: {baseline_accuracy:.2f}%", flush=True)
    print(f"[METRIC]\tBaseline mean bound: {baseline_mean_bound:.2f}", flush=True)

    indices = rng.choice(NUM_TRAIN, SUBSET_SIZE, replace=False)
    x_train_subset = x_train[indices]
    y_train_onehot_subset = y_train_onehot[indices]


    # NOTE: here's where you group your initial hyperparameter values and set boundaries on them
    initial_theta = [initial_s, initial_bandwidth, initial_sigma_n] 
    theta_range = [(0.9, 1.1), (6., 10.), (0.1, 5.)]
    # --- --- ---

    # computing log marginal likelihood 
    print("[LOG] Computing log marginal likehihood", flush=True)
    initial_lml = lml(x_train_subset, y_train_onehot_subset, initial_bandwidth, K_rbf, {'s': initial_s}, initial_sigma_n)
    
    
    print(f"[METRIC] Initial hyperparamters", flush=True)
    print(f"\tInitial bandwidth: {initial_bandwidth}", flush=True)
    print(f"\tInitial s: {initial_s}", flush=True)
    # print(f"\tOptimal lipschitz constant: {lipschitz}", flush=True)
    print(f"[METRIC]\tInitial log marginal likelihood: {initial_lml}", flush=True)


    print("[LOG] Optimising paramters", flush=True)
    opt_params, log_ml = optimize_theta(
        x_train_subset, y_train_onehot_subset, K_rbf, initial_theta, theta_range
    )
    
    optimal_s, optimal_bandwidth, optimal_sigma_n = opt_params # NOTE: unpack optimised parameters

    print(f"[METRIC] Optimised hyperparamters", flush=True)
    print(f"\tOptimal bandwidth: {optimal_bandwidth}", flush=True)
    print(f"\tOptimal s: {optimal_s}", flush=True)
    print(f"\tOptimal sigma_n: {optimal_sigma_n}", flush=True)
    # print(f"\tOptimal lipschitz constant: {optimal_lipschitz}", flush=True)
    print(f"[METRIC]\tOptimised marginal likelihood: {log_ml}", flush=True)

    print("[LOG] Creating a classifier with the optimised parameters", flush=True)
    nwc_optimised = NadarayaWatsonClassifier(
        bandwidth=optimal_bandwidth, 
        lipschitz_constant=lipschitz, 
        noise_parameter=sigma, 
        ck=1.,
        kernel=K_rbf, 
        kernel_kwargs={'s': optimal_s}
    )

    print("[LOG] Evaluating the optimised classifier", flush=True)
    optimized_predictions = []
    optimized_probabilities = []
    optimized_certainties = []

    print("[LOG] Running predictions with the optimised classifier", flush=True)
    for i in range(NUM_TEST):
        probs, bounds = nwc_optimised.predict(x_test[i], x_train, y_train_onehot)
        bounds = min(bounds, 1.)
        predicted_class = np.argmax(probs)
        optimized_predictions.append(predicted_class)
        optimized_probabilities.append(probs)
        optimized_certainties.append(bounds)

    optimized_accuracy = sum(optimized_predictions[i] == y_test[i] for i in range(NUM_TEST)) / NUM_TEST * 100
    # optimised_mean_bound = np.mean([np.mean(bounds) for bounds in optimised_certainties])
    optimized_mean_bound = np.mean(np.array(optimized_certainties))
    print(f"[METRIC]\tOptimised accuracy: {optimized_accuracy:.2f}% (baseline: {baseline_accuracy}%)", flush=True)
    print(f"[METRIC]\tOptimised mean bound: {optimized_mean_bound:.2f} (baseline: {baseline_mean_bound})", flush=True)

    K_initial = kernel_matrix(x_train_subset, initial_bandwidth, K_rbf, {'s': initial_s})
    K_optimal = kernel_matrix(x_train_subset, optimal_bandwidth, K_rbf, {'s': optimal_s})

    print("[LOG] Visualising results", flush=True)
    plt.figure(figsize=(12, 5))

    # Accuracy comparison
    plt.subplot(121)
    plt.bar(['Baseline', 'Optimised'], [baseline_accuracy, optimized_accuracy])
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)

    # Certainty comparison (mean bounds)
    plt.subplot(122)
    plt.bar(['Baseline', 'Optimised'], [baseline_mean_bound, optimized_mean_bound])
    plt.title('Mean Certainty Bounds')
    plt.ylabel('Bound')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title("Kernel Matrix with Initial Params")
    
    plt.imshow(K_initial)
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(122)
    plt.title("Kernel Matrix with Optimized Params")
    plt.imshow(K_optimal)
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('./outputs/figures/optimization_results.png')
    # print("[LOG] Results visualisation saved to './outputs/optimization_results.png'", flush=True)

if __name__ == '__main__':
    main()