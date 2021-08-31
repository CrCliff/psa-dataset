import numpy as np

def one_hot_encoding(classes):
    # Adapted from https://stackoverflow.com/a/58676802
    unique, inverse = np.unique(classes, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot