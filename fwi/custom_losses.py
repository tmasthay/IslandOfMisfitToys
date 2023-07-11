import torch
import torch.nn.functional as F

class W1(torch.nn.Module):
    def __init__(self):
        super(W1, self).__init__()

    def forward(self, y_pred, y_true):
        # Square the values
        y_pred = y_pred**2
        y_true = y_true**2

        # Calculate trapezoidal numerical integration
        y_pred_integral = torch.trapz(y_pred, dim=-1)
        y_true_integral = torch.trapz(y_true, dim=-1)

        # Divide by integral
        y_pred = y_pred / y_pred_integral
        y_true = y_true / y_true_integral

        # Calculate cumulative sum
        y_pred = torch.cumsum(y_pred, dim=-1)
        y_true = torch.cumsum(y_true, dim=-1)

        # Calculate squared difference and integrate
        squared_diff = torch.abs(y_pred - y_true)
        loss = torch.trapz(squared_diff, dim=-1)

        return loss
    
class W2(torch.nn.Module):
    def __init__(self):
        super(W1, self).__init__()

    def forward(self, y_pred, y_true):
        # Square the values
        y_pred = y_pred**2
        y_true = y_true**2

        # Calculate trapezoidal numerical integration
        y_pred_integral = torch.trapz(y_pred, dim=-1)
        y_true_integral = torch.trapz(y_true, dim=-1)

        # Divide by integral
        y_pred = y_pred / y_pred_integral
        y_true = y_true / y_true_integral

        # Calculate cumulative sum
        y_pred = torch.cumsum(y_pred, dim=-1)
        y_true = torch.cumsum(y_true, dim=-1)

        # Calculate squared difference and integrate
        squared_diff = (y_pred - y_true)**2
        loss = torch.trapz(squared_diff, dim=-1)

        return loss

