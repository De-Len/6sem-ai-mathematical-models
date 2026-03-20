import numpy as np


def z_score(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std