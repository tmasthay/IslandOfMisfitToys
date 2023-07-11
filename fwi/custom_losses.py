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
        super(W2, self).__init__()

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

class Huber(torch.nn.Module):
    def __init__(self, delta=0.5):
        super(Huber, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        # Calculate the absolute difference between the two seismic images
        diff = torch.abs(y_pred - y_true)

        # Apply the Huber loss function to each difference value
        square_h = 0.5 * (y_pred - y_true)**2
        linear_h = self.delta * torch.abs(y_pred - y_true) - 0.5 * self.delta**2
        h = torch.where(diff <= self.delta, square_h, linear_h )

         # Average over all elements in the images
        loss = torch.sum(h) / torch.numel(y_pred)

        return loss

class Hybrid_norm(torch.nn.Module):
    def __init__(self, delta=10.0):
        super(Hybrid_norm, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):  
        # Calculate the difference between the two seismic images
        r = y_pred - y_true
    
        # Apply the norm f(r) = sqrt(1 + (r/delta)^2) - 1
        h = torch.sqrt(1 + (r/self.delta)**2) - 1
        loss = torch.sum(h)

        return loss
