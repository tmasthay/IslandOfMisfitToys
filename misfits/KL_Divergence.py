import numpy as np

def KL_Divergence(f, g):
    # Ensure that the input signals have the same shape
    # Normalize the input signals
    f_normalized = f / np.sum(f)
    g_normalized = g / np.sum(g)

    # Calculate the KL divergence
    kl_div = np.sum(f_normalized * np.log(f_normalized / g_normalized))

    return kl_div