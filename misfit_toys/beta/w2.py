import torch
from returns.curry import curry

from misfit_toys.beta.prob import cdf, get_quantile_lambda, pdf


@curry
def w2(f, *, renorm, x, p, tol=1.0e-04, max_iters=20):
    fr = renorm(f, x)
    Q = get_quantile_lambda(
        fr, x, p=p, renorm=renorm, tol=tol, max_iters=max_iters
    )

    def helper(g, renorm_func=renorm):
        # print('PDF...', end='', flush=True)
        # tmp = pdf(g, x, renorm=renorm_func, dim=-1)
        # tmp = g / torch.trapz(g, x, dim=-1)
        tmp = renorm_func(g, x)
        tmp = tmp / torch.trapz(tmp, x, dim=-1)
        CDF = cdf(tmp, x, dim=-1)
        T = Q(CDF, deriv=False) - x
        integrand = T**2 * tmp
        res = torch.trapz(integrand, x, dim=-1)
        return res.sum()

    return helper, Q
