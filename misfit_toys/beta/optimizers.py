"""
Small module for wrapping common PyTorch optimizers.
"""

import torch


def get_optimizer(*, name, **kw):
    """
    Get an optimizer based on the specified name.

    Args:
        name (str): The name of the optimizer. Supported values are 'adam', 'sgd', and 'lbfgs'.
        **kw: Additional keyword arguments to be passed to the optimizer constructor.

    Returns:
        list: A list containing the optimizer class and the keyword arguments.

    Raises:
        ValueError: If the specified optimizer name is unknown.
    """
    if name == 'adam':
        return [torch.optim.Adam, kw]
    elif name == 'sgd':
        return [torch.optim.SGD, kw]
    elif name == 'lbfgs':
        return [torch.optim.LBFGS, kw]
    else:
        raise ValueError(f'Unknown optimizer: {name}')
