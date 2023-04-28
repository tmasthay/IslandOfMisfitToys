import numpy as np
#def func1():
#    print('func1')


def w1(f, g, dt): 
	"""
	Python code to compute the Wasserstein-1 distance between two one-dimensional probability distributions using the cumulative distribution functions.
	Parameters
    	----------
	f: array
     g: array
        f and g are the probability distributions to compare
	dt: int
	   time interval 
    	Returns
    	-------
    	float
        w1 norm of f.
	"""
	F = np.cumsum(f)
	F /= F[-1]
	G = np.cumsum(g)
	G /= G[-1]
# inverse
	w1 = dt * np.sum(np.abs(F-G))
	return w1
