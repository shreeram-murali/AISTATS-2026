import numpy as np
from typing import Callable, Optional, Dict, Any
from sklearn.metrics.pairwise import rbf_kernel

class LocalPolynomialClassifier:
    def __init__(self, bandwidth: float, degree: int = 2, reg_param: float = 1e-4, 
                 kernel: Optional[Callable] = rbf_kernel, kernel_kwargs: Optional[Dict[str, Any]] = None):
        self.lamda = bandwidth
        self.degree = degree
        self.reg = reg_param
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs or {}
        self.LOGS = dict()

    def _construct_design_matrix(self, X_diff):
        n_samples, n_features = X_diff.shape
        Z_components = [np.ones((n_samples, 1))]
        if self.degree >= 1:
            Z_components.append(X_diff)
        if self.degree >= 2:
            Z_components.append(X_diff ** 2)
        return np.hstack(Z_components)

    def _normalize_probs(self, raw_predictions):
        probs = np.maximum(raw_predictions, 0)
        total = np.sum(probs)
        if total > 0:
            probs = probs / total
        else:
            n_classes = raw_predictions.shape[0]
            probs = np.full(n_classes, 1.0 / n_classes)
        return probs

    def predict(self, x, X, Y):
        distances = np.linalg.norm(X - x, axis=1)
        weights = self.kernel(distances.reshape(-1, 1) / self.lamda, **self.kernel_kwargs).flatten()
        X_diff = X - x
        Z = self._construct_design_matrix(X_diff)
        Z_T_W = Z.T * weights
        A = Z_T_W @ Z
        A[np.diag_indices_from(A)] += self.reg
        B = Z_T_W @ Y
        try:
            beta = np.linalg.solve(A, B)
            raw_prediction = beta[0]
        except np.linalg.LinAlgError:
            n_classes = Y.shape[1]
            raw_prediction = np.full(n_classes, 1.0 / n_classes)
        probabilities = self._normalize_probs(raw_prediction)
        return probabilities, None

    def predict_proba(self, X, X_train, Y_train):
        probabilities = []
        for i in range(X.shape[0]):
            probs, _ = self.predict(X[i], X_train, Y_train)
            probabilities.append(probs)
        return np.array(probabilities)

    def predict_classes(self, X, X_train, Y_train):
        probabilities = self.predict_proba(X, X_train, Y_train)
        return np.argmax(probabilities, axis=1)