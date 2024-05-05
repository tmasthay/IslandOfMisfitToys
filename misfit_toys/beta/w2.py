import torch
from mh.core import draise
from returns.curry import curry
from torchcubicspline import natural_cubic_spline_coeffs as ncs, NaturalCubicSpline as NCS
import os
from mh.core import torch_stats
import torch.nn.functional as F
from time import time

torch.set_printoptions(callback=torch_stats())

def simple_coeffs(t, x):
    coeffs = ncs(t, x)
    right = F.pad(torch.stack([e.squeeze() for e in coeffs[1:]], dim=0), (1,0))
    return torch.cat([coeffs[0][None, :], right], dim=0)

def unwrap_coeffs(coeffs):
    return [e if i == 0 else e[1:].unsqueeze(-1) for i,e in enumerate(coeffs)]

def quantile_spline_coeffs(*, input_path, output_path, t, report_time=False):
    start_time = time()
    if os.path.exists(output_path):
        if( report_time ):
            v = torch.load(output_path)
            print(f"time: {time() - start_time}")
            return v
        return torch.load(output_path)
    u = torch.load(input_path)
    
    u_flatten = u.reshape(-1, u.shape[-1], 1)

    v = torch.empty(u_flatten.shape[0], t.shape[0], 5)
    for i in range(u_flatten.shape[0]):
        if i % 100 == 0:
            print(f"{i} / {u_flatten.shape[0]}: {time() - start_time}")
        v[i] = simple_coeffs(t, u_flatten[i]).T
    if report_time:
        print(f"time: {time() - start_time}")

    # double check this reshaping preserves the correct order
    v = v.permute((0, 2, 1)).reshape(*u.shape[:-1], 5, u.shape[-1])

    torch.save(v, output_path)
    return v


def main():
    input_path = "/home/tyler/miniconda3/envs/dw/data/marmousi/obs_data.pt"
    output_path = "out.pt"
    t = torch.linspace(0, 1, 750)
    v = quantile_spline_coeffs(input_path=input_path, output_path=output_path, t=t, report_time=True)
    print(v)

if __name__ == "__main__":
    main()


    
