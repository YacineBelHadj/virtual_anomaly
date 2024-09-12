import torch
from torch import nn


class FloodSignal(nn.Module):
    """ 
    A module to add a noise to a signal. 
    the noise point is to flood the signal meaning that the noise is added to the signal only if the signal is less than the noise.
    This similar to the effect of noise on Power spectral density of a signal.
    """
    def __init__(self,noise_level):
        super(FloodSignal,self).__init__()
        self.noise_level = noise_level
    def forward(self,signal):
        min_signal = torch.min(signal)
        min_signal = min_signal * (1 - torch.log10(self.noise_level))
        noise = torch.normal(min_signal,0.2,signal.shape)
        flodded_signal = torch.maximum(signal,noise)
        return flodded_signal
