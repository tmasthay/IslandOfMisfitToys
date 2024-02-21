# import os

# import hydra
# import matplotlib.pyplot as plt
# import torch
# from mh.core import DotDict, convert_dictconfig
# from mh.typlotlib import get_frames_bool, save_frames
# from returns.curry import curry

# from misfit_toys.fwi.loss.w2 import W2LossConst, cum_trap, unbatch_spline_eval
# from misfit_toys.utils import bool_slice, clean_idx


# def plotter(*, data, idx, fig, axes, cfg):
#     def lplot(title, xlabel, ylabel, d=None):
#         plt.title(f'{title} idx={clean_idx(idx)}')
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         if cfg.plt.static_ylim and d is not None:
#             plt.ylim(d.min().item(), d.max().item())

#     plt.clf()
#     plt.subplot(*cfg.plt.shape, 1)
#     plt.imshow(cfg.scaled_raw[idx[0], :, :].T, **cfg.plt.imshow_kw)
#     plt.plot(idx[1], data.shape[1] // 2, 'r*')
#     lplot('Depth-scaled obs data', 'Time Index', 'Receiver Index')
#     plt.colorbar()

#     plt.subplot(*cfg.plt.shape, 2)
#     plt.plot(cfg.t, data[idx], **cfg.plt.trace_kw)
#     lplot('Obs data trace', 'Time', 'Amplitude', data[idx[0], :, :])

#     plt.subplot(*cfg.plt.shape, 3)
#     plt.imshow(cfg.scaled_renorm[idx[0], :, :].T, **cfg.plt.imshow_kw)
#     plt.plot(idx[1], data.shape[1] // 2, 'r*')
#     lplot('Depth-scaled renormed data', 'Time Index', 'Receiver Index')
#     plt.colorbar()

#     plt.subplot(*cfg.plt.shape, 4)
#     plt.plot(cfg.t, cfg.renormed_data[idx], **cfg.plt.trace_kw)
#     lplot(
#         'Renormed data trace',
#         'Time',
#         'Amplitude',
#         cfg.renormed_data[idx[0], :, :],
#     )

#     plt.subplot(*cfg.plt.shape, 5)
#     plt.plot(cfg.p, cfg.q[idx], **cfg.plt.trace_kw)
#     lplot('Quantiles', 'p', 't', cfg.q[idx[0], :, :])

#     plt.subplot(*cfg.plt.shape, 6)
#     plt.plot(cfg.t, cfg.cdfs[idx], **cfg.plt.trace_kw)
#     lplot('CDFs', 'Time', 'Amplitude', cfg.cdfs[idx[0], :, :])

#     if cfg.plt.static_ylim:
#         plt.ylim(cfg.cdfs[idx[0], :, :].min().item(), cfg.cdfs.max().item())

#     return {'cfg': cfg}


# def go(cfg):
#     tensors = DotDict(
#         {
#             e.replace('.pt', ''): torch.load(os.path.join(cfg.data_path, e))
#             for e in os.listdir(cfg.data_path)
#             if e.endswith('.pt')
#         }
#     )
#     # tensors.obs_data = tensors.obs_data.permute(0, 2, 1)

#     # def renorm(x, eps=cfg.eps):
#     #     u = x**2 + eps
#     #     c = torch.trapz(u, dx=cfg.dt, dim=1)
#     #     return u / c.unsqueeze(1)

#     def renorm(x, eps=cfg.eps):
#         u = torch.abs(x) + eps
#         c = torch.trapz(u, dx=cfg.dt, dim=-1)
#         return u / c.unsqueeze(-1)

#     # cfg.renormed_data = renorm(x=tensors.obs_data, eps=cfg.eps)

#     cfg.t = torch.linspace(0.0, cfg.dt * (cfg.nt - 1), cfg.nt)

#     cfg.p = torch.linspace(0.0, 1.0, cfg.nt)
#     cfg.loss_fn = W2LossConst(
#         t=cfg.t, renorm=renorm, obs_data=tensors.obs_data, p=cfg.p
#     )
#     cfg.renormed_data = cfg.loss_fn.renorm_obs_data

#     cfg.t_scale = 1.0 + cfg.t_alpha * (cfg.t / cfg.t[-1]) ** cfg.t_pow
#     cfg.t_scale = cfg.t_scale.unsqueeze(0).unsqueeze(0)
#     cfg.scaled_raw = tensors.obs_data * cfg.t_scale
#     cfg.scaled_renorm = cfg.renormed_data * cfg.t_scale

#     cfg.q = unbatch_spline_eval(
#         cfg.loss_fn.quantiles, cfg.p.expand(*cfg.loss_fn.quantiles.shape, -1)
#     )
#     cfg.cdfs = cum_trap(cfg.renormed_data, dx=cfg.dt, dim=-1)

#     fig, axes = plt.subplots(*cfg.plt.shape, figsize=cfg.plt.figsize)
#     plt.subplots_adjust(**cfg.plt.subplots_adjust_kw)
#     iter = bool_slice(
#         *tensors.obs_data.shape,
#         none_dims=(2,),
#         ctrl=(lambda x, y: True),
#         start=(cfg.plt.start_shot, cfg.plt.start_rec, 0),
#         cut=(cfg.plt.cut_shot, cfg.plt.cut_rec, 0),
#         strides=(cfg.plt.stride_shot, cfg.plt.stride_rec, 1),
#     )
#     frames = get_frames_bool(
#         data=tensors.obs_data,
#         iter=iter,
#         fig=fig,
#         axes=axes,
#         plotter=plotter,
#         cfg=cfg,
#     )
#     save_frames(frames, path=cfg.plt.name, duration=cfg.plt.duration)


# @hydra.main(config_path='conf', config_name='debug', version_base=None)
# def main(cfg):
#     cfg = convert_config_simplest(cfg)
#     go(cfg)


# if __name__ == '__main__':
#     main()
