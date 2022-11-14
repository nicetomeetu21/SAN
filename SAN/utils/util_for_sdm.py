import torch
def approximation_heaviside_function(self, x):
    return torch.sigmoid(x*self.k)