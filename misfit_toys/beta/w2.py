import torch
from returns.curry import curry

from misfit_toys.beta.prob import cdf, get_quantile_lambda, pdf

from mh.core import draise

# @curry
# def w2(f, *, renorm, x, p, tol=1.0e-04, max_iters=20):
#     fr = renorm(f, x)
#     # raise ValueError(fr.shape)
#     Q = get_quantile_lambda(
#         frkz, x, p=p, renorm=renorm, tol=tol, max_iters=max_iters
#     )
#     # print('Q formed', flush=True)
#     # raise ValueError(type(Q))

#     def helper(g, *, renorm_func=renorm, lcl_Q=Q, lcl_x=x):
#         # print('PDF...', end='', flush=True)
#         # tmp = pdf(g, x, renorm=renorm_func, dim=-1)
#         # tmp = g / torch.trapz(g, x, dim=-1)
#         print('here', flush=True)
#         tmp = renorm_func(g, lcl_x)
#         tmp = tmp / torch.trapz(tmp, lcl_x, dim=-1)
#         CDF = cdf(tmp, lcl_x, dim=-1)
#         T = lcl_Q(CDF, deriv=False) - lcl_x
#         integrand = T**2 * tmp
#         res = torch.trapz(integrand, lcl_x, dim=-1)
#         return res.sum()

#     # return helper, Q
#     return helper


def w2(f, *, renorm, x, p, tol=1.0e-03, max_iters=20):
    fr = renorm(f, x)
    Q = get_quantile_lambda(
        fr, x, p=p, renorm=renorm, tol=tol, max_iters=max_iters
    )

    def helper(g, *, renorm_func=renorm, lcl_Q=Q, lcl_x=x):
        tmp = renorm_func(g, lcl_x)
        tmp2 = tmp / torch.trapz(tmp, lcl_x, dim=-1).unsqueeze(-1)
        CDF = cdf(tmp2, lcl_x, dim=-1)
        T = lcl_Q(CDF, deriv=False) - lcl_x
        integrand = T**2 * tmp
        res = torch.trapz(integrand, lcl_x, dim=-1)
        return res.sum()

    return helper
