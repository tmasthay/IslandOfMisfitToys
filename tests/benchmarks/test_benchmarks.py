from misfit_toys.tccs.modules.validate import main
import os
from masthay_helpers.global_helpers import DotDict


def test_validate():
    curr_dir = os.path.dirname(__file__)
    args = DotDict(
        {
            "output": os.path.join(curr_dir, "out", "validate.out"),
            "justify": "right",
            "clean": 'i',
        }
    )
    res = main(args)
    tol = 0.2
    diff = max([max(v) for v in res.values()])
    assert diff < tol, f"MARMOUSI TEST: diff={diff} > tol={tol}"


if __name__ == "__main__":
    test_validate()
