import torch


def get_optimizer(*, name, **kw):
    if name == 'adam':
        return [torch.optim.Adam, kw]
    elif name == 'sgd':
        return [torch.optim.SGD, kw]
    elif name == 'lbfgs':
        return [torch.optim.LBFGS, kw]
    else:
        raise ValueError(f'Unknown optimizer: {name}')
