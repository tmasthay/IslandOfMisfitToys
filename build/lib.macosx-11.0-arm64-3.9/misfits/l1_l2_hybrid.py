import numpy as np
import pytest

def Hibrid_norm(f,g, delta):
    """
    Computer the hibrid l1/l2 norm f(r) = sqrt(1 + (r/delta)^2) - 1 between two seismic images f and g with tuning parameter delta where r is defined as r = f - g
    Parameters
    ----------
    f, g: function
          f and g are the probability distributions to compare
    delta: float
      tunning parameter
    Returns
    -------
    float
    Hibrid l1/l2 norm
    """
    # Calculate the difference between the two seismic images
    r = f - g
    
    # Apply the norm f(r) = sqrt(1 + (r/delta)^2) - 1
    h = np.sqrt(1 + (r/delta)**2) - 1
    
    return h

