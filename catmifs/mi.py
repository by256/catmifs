"""
Methods for calculating exact Mutual Information for categorical distributions
in an embarrassingly parallel way.

Author: Batuhan Yildirim <by256@cam.ac.uk>
License: BSD 3 clause
"""

import numpy as np
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed


def _get_feature_feature_mi(x1, x2):
    # calculate entropy of x_1
    p_x1 = _marginal_probability(x1)
    H_x1 = _entropy(p_x1)
    # calculate entropy of x_1
    p_x2 = _marginal_probability(x2)
    H_x2 = _entropy(p_x2)
    # calculate joint entropy
    p_xx = _joint_probability(x1, x2)
    H_xx = _entropy(p_xx)
    return H_x1 + H_x2 - H_xx

def get_feature_feature_mi(X, F_idxs, S_idx, n_jobs):
    return Parallel(n_jobs=n_jobs)(delayed(_get_feature_feature_mi)(X[:, F_idx], X[:, S_idx]) for F_idx in F_idxs)

def _get_feature_target_mi(x, y, H_y):
    # calculate entropy of x
    p_x = _marginal_probability(x)
    H_x = _entropy(p_x)
    # calculate joint entropy
    p_xy = _joint_probability(x, y)
    H_xy = _entropy(p_xy)
    return H_x + H_y - H_xy

def get_feature_target_mi(X, y, n_jobs):
    n, p = X.shape
    H_y = _entropy_from_features(y)
    return Parallel(n_jobs=n_jobs)(delayed(_get_feature_target_mi)(X[:, i], y, H_y) for i in range(p))

def _entropy_from_features(x):
    p = _marginal_probability(x)
    return _entropy(p)

def _entropy(p, eps=1e-12):
    p = p + eps
    return -np.sum(p * np.log(p))

def _marginal_probability(x):
    unique, counts = np.unique(x, return_counts=True)
    return counts / np.sum(counts)

def _joint_probability(x, y):
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    n_joint = len(x_unique) * len(y_unique)
    xy = np.stack([x, y], axis=-1)
    joint_probs = []  # p(x, y)
    for x_val in x_unique:
        for y_val in y_unique:
            subset = xy[(xy[:, 0] == x_val) & (xy[:, 1] == y_val)]
            p_xy = len(subset) / len(xy)
            joint_probs.append(p_xy)
    return np.array(joint_probs)
