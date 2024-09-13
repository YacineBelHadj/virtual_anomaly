import torch
from torch import nn

class FloodSignal(nn.Module):
    """
    A module to flood a signal with noise. The noise is only added where the signal is lower than the noise level,
    simulating the effect of noise on the power spectral density of the signal.

    Attributes:
        noise_level (float): The noise level used to flood the signal.
        noise_std (float): The standard deviation of the noise applied.
    """
    
    def __init__(self, noise_level: float, noise_std: float = 0.3):
        """
        Initialize the FloodSignal module.

        Args:
            noise_level (float): The threshold level of noise.
            noise_std (float): Standard deviation of the noise to be applied.
        """
        super().__init__()
        self.noise_level = noise_level
        self.noise_std = noise_std

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to apply noise flooding to the signal.

        Args:
            signal (torch.Tensor): The input signal to be flooded with noise.

        Returns:
            torch.Tensor: The signal after applying noise flooding.
        """
        # Create noise based on the noise level and standard deviation
        noise_min = torch.min(signal) + self.noise_level 
        noise = torch.normal(mean=noise_min, std=self.noise_std, size=signal.size())
        noise = torch.abs(noise)
        noise = torch.clamp(noise, min=noise_min)

        # Apply the noise only where the signal is below the noise level
        flooded_signal = torch.where(signal < noise, noise, signal)

        return flooded_signal
