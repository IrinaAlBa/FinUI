import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree


def get_radius_kneighbors(x, n_neighbors):
    kd = KDTree(x, metric="chebyshev")
    neigh_dist = kd.query(x, k=n_neighbors+1)[0]
    
    return np.nextafter(neigh_dist[:, -1], 0)


def num_points_within_radius(x, radius):
    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    
    return np.array(nx) - 1.0


def preprocess_data(x):
    x = np.array(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError(f'x.ndim = {x.ndim}, should be 1 or 2')

    means = np.maximum(1e-100, np.mean(np.abs(x), axis=0))

    return (1/means) * x


def compute_mi(x, y, n_neighbors=5):
    # Kraskov
    n_samples = len(x)
    x, y = [preprocess_data(t) for t in [x, y]]
    xy = np.hstack((x, y))
    k = np.full(n_samples, n_neighbors)
    radius = get_radius_kneighbors(xy, n_neighbors)

    mask = (radius == 0)
    if mask.sum() > 0:
        vals, ix, counts = np.unique(
            xy[mask], axis=0, return_inverse=True, return_counts=True
        )
        k[mask] = counts[ix] - 1

    nx = num_points_within_radius(x, radius)
    ny = num_points_within_radius(y, radius)

    mi = max(0, digamma(n_samples) + np.mean(digamma(k))
             - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))
    return mi

def greedy(x, y, n_neighbors=5):
    idx = []
    rem = np.arange(0, x.shape[1])
    score = 0
    j = -1
    while len(rem)>0:
        mi = np.array([compute_mi(x[:, idx+[i]], y, n_neighbors) for i in rem])
        j = rem[np.argmax(mi)]
        mi = np.max(mi)
        if mi > score:
            score = mi
            rem = np.delete(rem, j)
            idx.append(j)
        else:
            break
        j = -1
    return idx