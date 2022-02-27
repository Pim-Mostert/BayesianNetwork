import numpy as np


def generate_random_probability_matrix(size):
    p = np.random.random(size)

    return p / p.sum(axis=-1, keepdims=True)