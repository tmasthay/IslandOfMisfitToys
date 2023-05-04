import numpy as np
import pytest

def hybrid_norm(f,g, delta):
    """
    Computer the hibrid l1/l2 norm J = sum f(r) where f(r) = sqrt(1 + (r/delta)^2) - 1 f and g are two seismic images, delta is a positive tuning parameter and r is defined as r = f - g
    Parameters
    ----------
    f, g: function
          f and g are the probability distributions to compare
    delta: float
          positive constant to be chosen
    Returns
    -------
    float
    Hibrid l1/l2 norm
    """

    # Calculate the difference between the two seismic images
    r = f - g
    
    # Apply the norm f(r) = sqrt(1 + (r/delta)^2) - 1
    h = np.sqrt(1 + (r/delta)**2) - 1
    
    return np.sum(h)

