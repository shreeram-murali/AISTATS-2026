import numpy as np
from typing import Callable, Optional, Dict, Any
from math import sqrt
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree

from sklearn.metrics.pairwise import rbf_kernel

class NadarayaWatsonClassifier():
    def __init__(self, bandwidth: float, lipschitz_constant: float, noise_parameter: float = 0.25, confidence: float = 0.05, 
                 ck: float = 1., kernel: Optional[Callable] = rbf_kernel, kernel_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialise an instance of Nadaraya-Watson (soft) Classifier
        Input: 
            Lamda: Bandwidth for kernels
            Lipschitz constant: || y - y' || <= L || x - x' || or 1/\gamma in case of separable distributions
            Noise parameter: \sigma <= 1/4
            Confidence: \delta = 0.05, implying 1-\delta confidence is the 95% range 
            Kernel: 
                - Kernel function where the parameter is the distance parameter divided by lamda
                - It's a callable, but you pass in || x - x' ||/lamda, along with other hyperparameters

        """

        if lipschitz_constant <= 0:
            raise ValueError("Lipschitz Constant must be a positive quantity.")

        self.lamda = bandwidth
        self.L = lipschitz_constant
        self.sigma = noise_parameter
        self.delta = confidence
        self.kernel = kernel
        self.ck = ck 
        self.kernel_kwargs = kernel_kwargs or {}
        
        self.LOGS = dict()

    
    def predict(self, x, X, Y):
        """
        Input:
            x: single test example (m-dimensional observation)
            X: training examples (n_samples x m observation dimensions)
            Y: one-hot encoded labels (n_samples x k label dimensions)
        Output:
            probabilities: array of probabilities, each element representing the probability that observation x belongs to the class at index of Y
            bound: 
        """
        distances = np.linalg.norm(X - x, axis=1)

        raw_weights = self.kernel(distances / self.lamda, **self.kernel_kwargs)
        weights = raw_weights / self.ck

        kappa_n = np.sum(weights)

        if kappa_n > 0:
            normalized_weights = weights / kappa_n
            normalized_weights = normalized_weights.reshape(-1, 1)

            probabilities = np.sum(normalized_weights * Y, axis=0)

        else:
            n_classes = Y.shape[1]
            probabilities = np.full(n_classes, 1.0 / n_classes)

        if kappa_n <= 1:
            alpha = np.sqrt(np.log(np.sqrt(2)/self.delta))
        else:
            alpha = np.sqrt(kappa_n * np.log((np.sqrt(1 + kappa_n)/self.delta)))

        self.LOGS.update(
            {'alpha': alpha,
             'kappa_n': kappa_n}
        )

        bound = self.L * self.lamda + (2 * self.sigma * alpha)/max(kappa_n, 1e-10)

        return probabilities, bound
    
    def logs(self):
        return self.LOGS
    
    def predict_proba(self, X, X_train, Y_train):
        probabilities = []
        for i in range(X.shape[0]):
            probs, _ = self.predict(X[i], X_train, Y_train)
            probabilities.append(probs)
        return np.array(probabilities)

    def predict_classes(self, X, X_train, Y_train):
        probabilities = self.predict_proba(X, X_train, Y_train)
        return np.argmax(probabilities, axis=1)

class DyadicNWC:    
    def __init__(self, m: int):
        if m < 1:
            raise ValueError("Resolution 'm' must be a positive integer.")
            
        self.m = m
        self.tree = {}
        self._dim = None
        self._n_classes = None
        self.LOGS = dict()
    
    def _get_cube_index(self, x: np.ndarray) -> tuple:
        divisions = 2**self.m
        indices = np.floor(x * divisions)
        indices = np.minimum(indices, divisions - 1)
        return tuple(indices.astype(int))
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Build the dyadic partition structure by preprocessing training data.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
                           Values must be scaled to [0, 1].
            Y (np.ndarray): One-hot encoded labels of shape (n_samples, n_classes).
        """
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("Input data X must be scaled to the range [0, 1].")
        
        self._dim = X.shape[1]
        self._n_classes = Y.shape[1]
        self.tree = {}
        
        for i in range(X.shape[0]):
            cube_idx = self._get_cube_index(X[i])
            
            if cube_idx not in self.tree:
                self.tree[cube_idx] = {
                    'label_sum': np.zeros(self._n_classes),
                    'count': 0
                }

            self.tree[cube_idx]['label_sum'] += Y[i]
            self.tree[cube_idx]['count'] += 1
            
        return self
    
    def predict(self, x: np.ndarray) -> tuple:
        """
        Predict class probabilities for a single test point.
        
        Args:
            x (np.ndarray): Single test example (m-dimensional).
            
        Returns:
            tuple: (probabilities, bound) where probabilities is array of class
                   probabilities and bound is the confidence bound.
        """
        if self.tree is None:
            raise RuntimeError("The model must be fitted before prediction.")
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("Input data x must be scaled to the range [0, 1].")
        
        cube_idx = self._get_cube_index(x)
        cube_data = self.tree.get(cube_idx)
        
        if cube_data is None or cube_data['count'] == 0:
            probabilities = np.full(self._n_classes, 1.0 / self._n_classes)
        else:
            probabilities = cube_data['label_sum'] / cube_data['count']
        
        return probabilities
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = []
        for i in range(X.shape[0]):
            probs = self.predict(X[i])
            probabilities.append(probs)
        return np.array(probabilities)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

class LocalizedNWC:
    def __init__(self, bandwidth: float, lipschitz_constant: float, noise_parameter: float = 0.25, confidence: float = 0.05, 
                 ck: float = 1., kernel: Optional[Callable] = rbf_kernel, kernel_kwargs: Optional[Dict[str, Any]] = None):
        
        if lipschitz_constant <= 0:
            raise ValueError("Lipschitz Constant must be a positive quantity.")

        self.lamda = bandwidth
        self.L = lipschitz_constant
        self.sigma = noise_parameter
        self.delta = confidence
        self.kernel = kernel
        self.ck = ck 
        self.kernel_kwargs = kernel_kwargs or {}
        
        self.tree = None
        self.X_train = None
        self.Y_train = None
        self.LOGS = dict()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X_train = np.asarray(X)
        self.Y_train = np.asarray(Y)
        self.tree = KDTree(self.X_train)
        return self

    def predict(self, x: np.ndarray, k: int):
        if self.tree is None:
            raise RuntimeError("The model must be fitted before making predictions.")

        x = np.asarray(x).reshape(1, -1)
        distances, indices = self.tree.query(x, k=k)
        
        distances = distances[0]
        indices = indices[0]

        weights = np.zeros(k)
        for i in range(k):
            raw_weight = self.kernel(distances[i] / self.lamda, **self.kernel_kwargs)
            weights[i] = raw_weight / self.ck

        kappa_n = np.sum(weights)
        if kappa_n > 0:
            weights = weights / kappa_n

        probabilities = np.zeros(self.Y_train.shape[1])
        for i in range(k):
            probabilities += weights[i] * self.Y_train[indices[i]]

        if kappa_n <= 1:
            alpha = np.sqrt(np.log(np.sqrt(2)/self.delta))
        else:
            alpha = np.sqrt(kappa_n * np.log((np.sqrt(1 + kappa_n)/self.delta)))

        self.LOGS.update({
            'alpha': alpha, 
            'kappa_n': kappa_n,
            'k_neighbors': k
        })

        bound = self.L * self.lamda + (2 * self.sigma * alpha)/max(kappa_n, 1e-10)

        return probabilities, bound

    def logs(self):
        return self.LOGS

    def predict_proba(self, X, Y, k):
        pass

    def predict_classes(self, X, k):
        pass


