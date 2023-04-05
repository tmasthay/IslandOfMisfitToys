import numpy as np

def sobolev_norm(f, g, dh, k=2):

    """
    Compute the Sobolev norm of the difference between two 1D functions f(x) and g(x)
    over the domain specified by the array `domain`, up to order k.
    
    Parameters
    ----------
    f : function
        A function that takes a single argument x and returns the value of f(x).
    g : function
        A function that takes a single argument x and returns the value of g(x).
    dh : int,
        The distance between two data points in function.
    k : int, optional
        The order of the Sobolev norm. Defaults to 2.
        
    Returns
    -------
    float
        The Sobolev norm of the difference between the two functions f(x) and g(x) up to order k.
    """
    f_values = np.array(f)
    g_values = np.array(g)
    
    # Compute the derivative of f(x) and g(x) up to order k using finite differences
    Df_values = [f_values]
    Dg_values = [g_values]
    for i in range(k):
        Df_values.append(np.gradient(Df_values[-1], dh))
        Dg_values.append(np.gradient(Dg_values[-1], dh))
    
    # Compute the Sobolev norm using the formula
    # ||f-g||_k^2 = ||f-g||_2^2 + sum_{i=1}^k ||D^if - D^ig||_2^2
    norm2 = np.sum((f_values - g_values)**2)
    normk2 = 0
    for i in range(1, k+1):
        normk2 += np.sum((Df_values[i] - Dg_values[i])**2)
    
    return np.sqrt(norm2 + normk2)
