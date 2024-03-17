import os

import pynvml as nvml
from mh.core import DotDict

import misfit_toys.examples.marmousi.validate as val


def check_gpu_memory():
    nvml.nvmlInit()
    deviceCount = nvml.nvmlDeviceGetCount()
    u = [0.0] * deviceCount
    for i in range(deviceCount):
        handle = nvml.nvmlDeviceGetHandleByIndex(i)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        u[i] = info.free
    nvml.nvmlShutdown()
    return u


def test_validate():
    free_gpu = check_gpu_memory()
    # set gpu need way higher than actual need to test failure case
    total_gpu_need = 150.0e9
    min_mem_per_gpu = total_gpu_need / len(free_gpu)
    if min(free_gpu) < min_mem_per_gpu:
        with open('tests/status/test_validate_gpu_saturation.txt', 'w'):
            pass
        assert False, (
            f"Error: Not enough resources. Need {min_mem_per_gpu / 1.0e9} GB"
            f" per GPU\nHave {free_gpu} GB on each GPU, respectively.\n"
        )

    curr_dir = os.path.dirname(__file__)
    args = DotDict(
        {
            "output": os.path.join(curr_dir, "out", "validate.out"),
            "justify": "right",
            "clean": 'i',
        }
    )
    res = val.main(args)
    tol = 0.2
    diff = max([max(v) for v in res.values()])
    assert diff < tol, f"MARMOUSI TEST: diff={diff} > tol={tol}"
    print(f'SUCCESS: {diff} < {tol}')


if __name__ == "__main__":
    test_validate()
