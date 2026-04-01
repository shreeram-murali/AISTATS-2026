import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict
import pandas as pd

from torchvision import datasets, transforms
from scipy import optimize
from sklearn.model_selection import train_test_split
from scipy.special import gamma, kv 
from math import pi as PI 

from reml import NadarayaWatsonClassifier


rng = np.random.default_rng(seed=None) 

CONFIG = {
    "data": {
        "num_train_total": 30000,
        "num_test_total": 500,
    },
    "optimization": {
        "subset_size": 3000,
        "validation_split_ratio": 0.1,
        "r_factor": 0.95, # between 0 and 1, a higher number means accuracy is weighted more than bounds 
        "initial_params": [0.75, 7.5],
        "param_bounds": [(0.6, 1.), (1., 20.)], 
        "maxiter": 10
    },
    "model": {
        "lipschitz": 0.02,
        "sigma": 0.25
    }
}

"""
KERNEL FUNCTIONS
Here's where all the kernel functions are defined. If you want a new one, add them here. 
"""
def K_rbf(x, s=1.):
    squared_exponential = s ** 2 * np.exp(-(x**2/2))    
    return 1e-8 if x > 1. else min(squared_exponential, 1.)

def K_boxcar(x, s=0.5):
    return min(1., s) if x <= 1. else 1e-8

def K_linear(x, s=0.5):
    linear = s if x == 0. else s * x 
    return 1e-8 if x > 1. else linear

def K_matern(x, s, v, l):
    if np.isscalar(x) and x == 0:
        x = 1e-8
    elif isinstance(x, np.ndarray):
        x = x.copy()
        x[x==0] = 1e-8
    
    part1 = 2 ** (1 - v) / gamma(v)
    part2 = (np.sqrt(2 * v) * np.abs(x) / l) ** v
    part3 = kv(v, np.sqrt(2 * v) * np.abs(x) / l)
    
    return s**2 * part1 * part2 * part3

def K_epanechnikov(x, s: float=0.75):
    epanechinkov = s * (1 - x**2)
    return 1e-8 if x > 1. else epanechinkov

def K_quartic(x, s: float=15/16):
    quartic = s * (1 - x**2)**2
    return 1e-8 if x > 1. else quartic

def K_triweight(x, s:float=35/32):
    triweight = s * (1 - x**2)**3
    return 1e-8 if x > 1. else triweight

def K_cosine(x, s: float = 1.):
    cosine = s * PI/4 * np.cos((PI/2) * x)
    return 1e-8 if x > 1. else cosine

def K_tricube(x, s: float = 70/81):
    tricube = s * (1 - abs(x)**3)**3
    return 1e-8 if x > 1. else tricube

def K_silverman(x, s=0.5):
    val = np.abs(x / np.sqrt(2))
    silverman =  s * np.exp(-val) * np.sin(val + PI / 4)
    return 1e-8 if x > 1. else silverman

"""
DATASET PREPARATION
"""
def load_and_prepare_data(config: Dict) -> Dict:
    print("[LOG] Loading and preparing data...")
    train_set = datasets.MNIST('./data', train=True, download=True)
    test_set = datasets.MNIST('./data', train=False, download=True)

    x_train_full = train_set.data.numpy().reshape(len(train_set), -1) / 255.0
    y_train_full_labels = train_set.targets.numpy()
    y_train_full_onehot = np.eye(10)[y_train_full_labels]
    
    x_test_final = test_set.data.numpy().reshape(len(test_set), -1) / 255.0
    y_test_final_labels = test_set.targets.numpy()
    
    test_indices = rng.choice(len(x_test_final), config['data']['num_test_total'], replace=False)
    x_test = x_test_final[test_indices]
    y_test = y_test_final_labels[test_indices]

    subset_indices = rng.choice(len(x_train_full), config['optimization']['subset_size'], replace=False)
    x_subset = x_train_full[subset_indices]
    y_subset_onehot = y_train_full_onehot[subset_indices]
    
    X_opt_train, X_opt_val, y_opt_train_onehot, y_opt_val_onehot = train_test_split(
        x_subset, y_subset_onehot,
        test_size=config['optimization']['validation_split_ratio']
    )
    y_opt_val = np.argmax(y_opt_val_onehot, axis=1)

    print(f"[LOG] Optimization will use {X_opt_train.shape[0]} samples for training and {X_opt_val.shape[0]} for validation.")

    dataset = {
        "full_train": (x_train_full[:config['data']['num_train_total']], y_train_full_onehot[:config['data']['num_train_total']]),
        "final_test": (x_test, y_test),
        "opt_train": (X_opt_train, y_opt_train_onehot),
        "opt_val": (X_opt_val, y_opt_val)
    }
    return dataset


def evaluate_model_performance(params: list, config: Dict, train_set: tuple, val_set: tuple) -> tuple:
    s, bandwidth = params
    X_train, y_train_onehot = train_set
    X_val, y_val = val_set

    nwc = NadarayaWatsonClassifier(
        bandwidth=bandwidth,
        lipschitz_constant=config['model']['lipschitz'],
        noise_parameter=config['model']['sigma'],
        ck=1.,
        kernel=K_epanechnikov,
        kernel_kwargs={'s': s}
    )

    predictions = []
    bounds_list = []
    
    for i in range(len(X_val)):
        probs, bounds = nwc.predict(X_val[i], X_train, y_train_onehot)
        bounds = min(bounds, 1.)
        predictions.append(np.argmax(probs))
        bounds_list.append(bounds)
    
    accuracy = np.mean(np.array(predictions) == y_val)
    mean_bound = np.mean(bounds_list) if bounds_list else 0.0
    
    return accuracy, mean_bound


def find_optimal_params(config: Dict, opt_datasets: Dict) -> tuple:
    """
    Runs the scipy optimizer to find the best hyperparameters based on the
    composite objective function.
    """

    PLOTTINGLOGS = []
    
    def objective(params: list) -> float:
        """The function to be minimized by scipy.optimize."""
        accuracy, mean_bound = evaluate_model_performance(
            params, config, opt_datasets['opt_train'], opt_datasets['opt_val']
        )

        r = config['optimization']['r_factor']
        # score = r * accuracy/(mean_bound + 1e-9)
        score = r * accuracy - ((1 - r) * mean_bound) # reward accuracy and penalise bound 

        print(f"[Optimizer] Trying s={params[0]:.1f}, bw={params[1]:.2f} | Acc: {accuracy:.3f}, Bnd: {mean_bound:.3f} | Score: {score:.4f}")

        PLOTTINGLOGS.append({
            's': params[0],
            'bw': params[1], 
            'acc': accuracy,
            'bnd': mean_bound,
            'score': score
        })
    
        return -score

    print("\n--- Starting Optimization ---")
    result = optimize.minimize(
        fun=objective,
        x0=config['optimization']['initial_params'],
        method='Powell',
        bounds=config['optimization']['param_bounds'], 
        options={'maxiter': CONFIG['optimization']['maxiter']}
    )

    df = pd.DataFrame(PLOTTINGLOGS)
    csv_filename = 'hyperparameter-optimization/outputs/logs/optimization_log_cosine_brrbrrpatapim.csv' #NOTE: change this to prevent overwriting! 
    # df.to_csv(csv_filename, index=False)
    print(f"[LOG] Optimization log saved to {csv_filename}")

    return result.x, -result.fun


def plot_results(baseline_metrics: dict, optimized_metrics: dict):
    #NOTE: this is google gemini written, need to move this stuff into main
    """Generates bar charts to compare baseline and optimized performance."""
    print("\n[LOG] Visualising results...")
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ['Baseline', 'Optimised']
    
    accuracies = [baseline_metrics['accuracy'] * 100, optimized_metrics['accuracy'] * 100]
    ax1.bar(labels, accuracies, color=['#4C72B0', '#55A868'])
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 1, f"{v:.2f}%", ha='center', color='black')

    bounds = [baseline_metrics['mean_bound'], optimized_metrics['mean_bound']]
    ax2.bar(labels, bounds, color=['#4C72B0', '#55A868'])
    ax2.set_title('Mean Certainty Bound Comparison')
    ax2.set_ylabel('Bound Value (Lower is Better)')
    
    ax2.set_ylim(0, max(bounds) * 1.2)
    for i, v in enumerate(bounds):
        ax2.text(i, v + (max(bounds) * 0.01), f"{v:.4f}", ha='center', color='black')
        
    plt.tight_layout()
    plt.show()


def main():
    
    dataset = load_and_prepare_data(CONFIG)
    
    print("\n--- Evaluating Baseline Performance ---")
    baseline_params = CONFIG['optimization']['initial_params']
    base_acc, base_bnd = evaluate_model_performance(
        baseline_params, CONFIG,
        train_set=dataset['full_train'],
        val_set=dataset['final_test']
    )
    baseline_metrics = {'accuracy': base_acc, 'mean_bound': base_bnd}
    print(f"[Result] Baseline Accuracy: {base_acc:.2%}, Baseline Mean Bound: {base_bnd:.4f}")

    optimal_params, best_score = find_optimal_params(CONFIG, dataset)
    
    print("\n--- Evaluating Optimized Model Performance ---")
    opt_acc, opt_bnd = evaluate_model_performance(
        optimal_params, CONFIG,
        train_set=dataset['full_train'],
        val_set=dataset['final_test']
    )
    optimized_metrics = {'accuracy': opt_acc, 'mean_bound': opt_bnd}
    
    print("\n" + "="*30)
    print("           FINAL RESULTS")
    print("="*30)
    print(f"Initial Params (s, bw): {baseline_params}")
    print(f"  -> Baseline Accuracy: {base_acc:.2%}")
    print(f"  -> Baseline Mean Bound: {base_bnd:.4f}\n")
    
    print(f"Optimized Params (s, bw): [{optimal_params[0]:.2f}, {optimal_params[1]:.2f}]")
    print(f"  -> Optimized Accuracy: {opt_acc:.2%}")
    print(f"  -> Optimized Mean Bound: {opt_bnd:.4f}")
    print("="*30)
    
    plot_results(baseline_metrics, optimized_metrics)

if __name__ == '__main__':
    main()