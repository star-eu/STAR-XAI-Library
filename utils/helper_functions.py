import numpy as np


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))
