"""Simulated recommendation environments built from MovieLens and Yelp.

In round t a new item arrives (a movie for MovieLens, a Yelp user for Yelp) and every
base arm (a MovieLens user, a Yelp restaurant) gets the context

    x_i = arm_features[i] * item_features[t + 1]        (element-wise product).

Base arm i is clicked with probability mean(x_i), and a super arm of K base arms earns
reward 1 if at least 80% of its base arms are clicked.
"""
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

DATA_DIR = Path(__file__).resolve().parent / "data"
MOVIELENS_CLUSTERS = 22


def expected_reward(contexts):
    """Click probability of each base arm: 2 * sigmoid(sum(x) / (2 * nnz(x))) - 1."""
    nonzero = np.maximum(np.count_nonzero(contexts, axis=1), 1)
    return 2 / (1 + np.exp(-contexts.sum(axis=1) / (2 * nonzero))) - 1


def super_arm_reward(base_rewards):
    """A super arm succeeds when at least 80% of its base arms are clicked."""
    return int(np.sum(base_rewards) >= 0.8 * len(base_rewards))


class Environment:
    def __init__(self, name, arm_features, item_features, cluster_labels=None):
        self.name = name
        self.arm_features = arm_features
        self.item_features = item_features
        self.num_arms, self.dim = arm_features.shape
        self._cluster_labels = cluster_labels

    @property
    def max_rounds(self):
        return len(self.item_features) - 1

    @property
    def clusters(self):
        """List of arrays with the base arm indices that belong to each cluster."""
        if self._cluster_labels is None:
            kmeans = KMeans(n_clusters=MOVIELENS_CLUSTERS, init="k-means++", max_iter=300,
                            n_init=10, random_state=0)
            self._cluster_labels = kmeans.fit_predict(self.arm_features)
        labels = self._cluster_labels
        return [np.where(labels == c)[0] for c in range(labels.max() + 1)]

    def contexts(self, t):
        return self.arm_features * self.item_features[t + 1]


def load_movielens():
    """Users are the base arms (20 genre preferences); each round a movie arrives."""
    data = np.load(DATA_DIR / "movielens.npz")
    return Environment("movielens", data["user_prefs"], data["movie_genres"].astype(np.float64))


def load_yelp():
    """Restaurants are the base arms (171 category features); each round a user arrives."""
    path = DATA_DIR / "yelp.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Yelp Open Dataset and run "
            "`python prepare_yelp.py /path/to/yelp_dataset` first (see README).")
    data = np.load(path)
    return Environment("yelp", data["restaurant_features"], data["user_prefs"], data["clusters"])


def load_environment(name):
    return {"movielens": load_movielens, "yelp": load_yelp}[name]()
