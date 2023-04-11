import numpy as np

def huber_norm(f, g, delta):
    """
    Computer the huber norm Huber_norm(f, g, delta) = (1/N) * sum_i=1^N h_delta(f_i - g_i)
    between two seismic images f and g with tuning parameter delta where delta is defined as
    h_delta(x) = {0.5 * x^2 if |x| <= delta
	          delta * |x| - 0.5 * delta^2 if |x| > delta}
    Parameters
    ----------
    f, g: function
          f and g are the probability distributions to compare
    delta: float
	  tunning parameter 
    Returns
    -------
    float
    Huber norm
    """
    # Calculate the absolute difference between the two seismic images
    diff = np.abs(f - g)

    # Apply the Huber loss function to each difference value
    h = np.zeros(f.shape)
    idx = (diff <= delta)
    h[idx] = 0.5 * diff[idx]**2
    idx = (diff > delta)
    h[idx] = delta * diff[idx] - 0.5 * delta**2

    # Average over all elements in the images
    return np.sum(h) / np.size(f)

