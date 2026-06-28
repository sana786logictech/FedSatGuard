
import numpy as np

class MahalanobisDetector:
    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])

    def score(self, x):
        d = x - self.mu
        return float(d.T @ np.linalg.inv(self.cov) @ d)
