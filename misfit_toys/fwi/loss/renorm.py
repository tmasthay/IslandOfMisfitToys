import torch


def square(x):
    """
    Squares each element of the input array and normalizes the result.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Array with squared and normalized values.
    """
    u = x**2
    return u / u.sum()


def shift(x, s=None):
    """
    Shifts the input array `x` by subtracting the minimum value `s` (or the minimum value of `x` if `s` is not provided).
    Returns the shifted array normalized by dividing each element by the sum of all elements in the shifted array.

    Args:
        x (numpy.ndarray): The input array.
        s (float, optional): The value to subtract from `x`. If not provided, the minimum value of `x` is used.

    Returns:
        numpy.ndarray: The shifted and normalized array.

    """
    if s is None:
        s = x.min()
    u = x - s
    return u / u.sum()


def exp(x, alpha=-1.0):
    """
    Compute the exponential function with an optional scaling factor.

    Args:
        x (torch.Tensor): The input tensor.
        alpha (float, optional): The scaling factor. Defaults to -1.0.

    Returns:
        torch.Tensor: The result of applying the exponential function to the input tensor.

    """
    u = torch.exp(alpha * x)
    return u / u.sum()
