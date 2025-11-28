import numpy as np
from sklearn.cluster import SpectralBiclustering
from functools import wraps
from time import time


def normalize(a, axis=None):
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap


@timing
def solve_spectral_coclustering(P, n_row_clusters, n_col_clusters):
    model = SpectralBiclustering(n_clusters=(n_row_clusters, n_col_clusters))
    model.fit(P)

    return model.row_labels_, model.column_labels_


def get_kl_cost(data_dist, centroids, epsilon=1e-12):
    log_centroids = np.log(centroids + epsilon)
    return -1 * np.dot(data_dist, log_centroids.T)


@timing
def solve_information_theoretic_coclustering(P, n_row_clusters, n_col_clusters, max_iter=20, epsilon=1e-12):
    P_joint = P + epsilon
    P_joint = P_joint / P_joint.sum()

    n, m = P_joint.shape

    np.random.seed(42)
    row_labels = np.random.randint(0, n_row_clusters, n)
    col_labels = np.random.randint(0, n_col_clusters, m)

    old_row_labels = np.copy(row_labels)
    old_col_labels = np.copy(col_labels)

    for it in range(max_iter):
        P_given_x_aggregated = np.zeros((n, n_col_clusters))
        for c in range(n_col_clusters):
            mask = (col_labels == c)
            if np.any(mask):
                P_given_x_aggregated[:, c] = P_joint[:, mask].sum(axis=1)

        row_sums = P_given_x_aggregated.sum(axis=1, keepdims=True) + epsilon
        P_given_x_aggregated /= row_sums

        row_centroids = np.zeros((n_row_clusters, n_col_clusters))
        for r in range(n_row_clusters):
            mask = (row_labels == r)
            if np.any(mask):
                row_centroids[r, :] = P_given_x_aggregated[mask].mean(axis=0)
            else:
                row_centroids[r, :] = np.random.rand(n_col_clusters)
                row_centroids[r, :] /= row_centroids[r, :].sum()

        cost_matrix_rows = get_kl_cost(P_given_x_aggregated, row_centroids)
        row_labels = np.argmin(cost_matrix_rows, axis=1)

        P_given_y_aggregated = np.zeros((m, n_row_clusters))
        for r in range(n_row_clusters):
            mask = (row_labels == r)
            if np.any(mask):
                P_given_y_aggregated[:, r] = P_joint[mask, :].sum(axis=0)

        col_sums = P_given_y_aggregated.sum(axis=1, keepdims=True) + epsilon
        P_given_y_aggregated /= col_sums

        col_centroids = np.zeros((n_col_clusters, n_row_clusters))
        for c in range(n_col_clusters):
            mask = (col_labels == c)
            if np.any(mask):
                col_centroids[c, :] = P_given_y_aggregated[mask].mean(axis=0)
            else:
                col_centroids[c, :] = np.random.rand(n_row_clusters)
                col_centroids[c, :] /= col_centroids[c, :].sum()

        cost_matrix_cols = get_kl_cost(P_given_y_aggregated, col_centroids)
        col_labels = np.argmin(cost_matrix_cols, axis=1)
        if np.array_equal(row_labels, old_row_labels) and np.array_equal(col_labels, old_col_labels):
            break

        old_row_labels = np.copy(row_labels)
        old_col_labels = np.copy(col_labels)

    return row_labels, col_labels
