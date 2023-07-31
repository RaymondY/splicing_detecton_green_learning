# Modified from https://github.com/yunchengwang/python-feature-test by Tiankai Yang

import numpy as np


class FeatureTest:
    def __init__(self):
        self.dim_loss = dict()
        self.sorted_features = None
        self.dim = 0

    def fit(self, X, y, n_bins, outliers=False):
        self.dim = X.shape[1]
        for d in range(self.dim):
            min_partition_loss = self.get_min_partition_loss(X[:, d], y, n_bins, outliers)
            self.dim_loss[d] = min_partition_loss

        self.dim_loss = {k: v for k, v in sorted(self.dim_loss.items(), key=lambda item: item[1])}
        self.sorted_features = np.array(list(self.dim_loss.keys()))

    def transform(self, X, n_selected):
        assert self.sorted_features is not None, f'Run fit() before selecting features.'
        assert X.shape[1] == self.dim, f'Expect feature dimension {self.dim}, but got {X.shape[1]}.'
        return X[:, self.sorted_features[np.arange(n_selected)]]

    def fit_transform(self, X, y, n_bins, n_selected):
        self.fit(X, y, n_bins)
        return self.transform(X, n_selected)

    def get_min_partition_loss(self, f_1d, y, n_bins, outliers=False):
        if outliers:
            f_1d, y = self.remove_outliers(f_1d, y)
        min_partition_loss = float('inf')
        f_min, f_max = f_1d.min(), f_1d.max()
        bin_width = (f_max - f_min) / n_bins
        for i in range(1, n_bins):
            partition_point = f_min + i * bin_width
            y_l, y_r = y[f_1d <= partition_point], y[f_1d > partition_point]
            partition_loss = self.get_loss(y_l, y_r)
            if partition_loss < min_partition_loss:
                min_partition_loss = partition_loss
        return min_partition_loss
    
    def get_loss(self, y_l, y_r):
        n1, n2 = len(y_l), len(y_r)
        left_mse = ((y_l - y_l.mean()) ** 2).sum()
        right_mse = ((y_r - y_r.mean()) ** 2).sum()
        return np.sqrt((left_mse + right_mse) / (n1 + n2))

    @staticmethod
    def remove_outliers(f_1d, y, n_std=2.0):
        """Remove outliers for the regression problem."""
        f_mean, f_std = f_1d.mean(), f_1d.std()
        return f_1d[np.abs(f_1d - f_mean) <= n_std * f_std], y[np.abs(f_1d - f_mean) <= n_std * f_std]
