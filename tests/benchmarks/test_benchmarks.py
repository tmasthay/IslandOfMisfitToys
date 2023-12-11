import misfit_toys.examples.marmousi.validate as val
import os
from masthay_helpers.global_helpers import DotDict

# import logging
# import pytest


def test_validate():
    # logging.basicConfig(
    #     level=logging.WARNING, filename='/tmp/pytest_warnings.log'
    # )
    # logging.captureWarnings(True)
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


if __name__ == "__main__":
    test_validate()
