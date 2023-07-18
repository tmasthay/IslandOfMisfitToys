import pytest
import torch
from acoustic_fwi import W1  # adjust this import as necessary
from unit_test_helpers import *

# Fixtures for the test functions
@pytest.fixture(
    params=[lambda x: x, lambda x: x**2],
    ids=['identity', 'square']
)
def renorm_func(request):
    return request.param

@pytest.fixture
def model(renorm_func):
    return W1(renorm_func)
    
# Test for prep_data function
@pytest.mark.parametrize(
    'y_true, expected',
    [
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            (lambda rf: \
                torch.tensor([
                    [1./6., 3./6., 1.], [4./15., 9./15., 1.]
                ]) if rf.__name__ == "identity" \
                else torch.tensor([
                    [1./46., 10./46., 1.], [4./322., 81./322., 1.]
                ])
            )(renorm_func)
        )
    ],
    ids=['simple_case_1']
)
def test_prep_data(model, y_true, expected):
    result = model.prep_data(y_true)
    assert check(result, expected)

# Test for forward function
@pytest.mark.parametrize(
    'y_true, y_pred, expected',
    [
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            0.0
        )
    ],
    ids=['simple_case_1']
)
def test_forward(model, y_true, y_pred, expected):
    result = model.forward(y_pred, y_true)
    assert check(result, expected)

