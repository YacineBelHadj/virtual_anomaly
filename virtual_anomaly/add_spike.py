import torch
from torch import nn
from scipy.signal import windows

class AddSpike(nn.Module):
    """
    A module to add or modify harmonic content in a 1D signal.

    This class is designed to facilitate the alteration of spectral content by adding
    a window (modulation) around a point of interest, which can then
    be used to simulate the addition or attenuation of harmonics in the signal.
    
    Attributes:
        data_axis (torch.Tensor): The axis representing the data (e.g., frequency or time).
        window_center (float): The point of interest around which the harmonic modification is applied.
        window_size (float): The width of the window around the point of interest.
        amplitude (float): The amplitude factor applied to the triangular window.
        modulation (torch.Tensor): The constructed window used to modify the signal.
    """

    def __init__(self, data_axis, window_center:float, window_size:float, amplitude:float):
        """
        Initialize the HarmonicModifier module.

        Args:
            data_axis (array-like): The axis representing the data (e.g., frequency or time).
            window_center (float): The point of interest around which the harmonic modification is applied.
            window_size (float): The width of the window around the point of interest.
            amplitude (float): The amplitude factor applied to the triangular window.
        """
        super().__init__()
        if isinstance(data_axis, torch.Tensor):
            data_axis = data_axis.clone().detach().float()  # Clone and detach if it's already a tensor
        else:
            data_axis = torch.tensor(data_axis, dtype=torch.float32)  # Ensure data_axis is a float32 tensor

        self.data_axis = data_axis
        self.window_center = window_center
        self.window_size = window_size
        self.amplitude = amplitude
        self.modulation = self.construct_modulation(data_axis, window_center, window_size, amplitude)

    def construct_modulation(self, data_axis, window_center, window_size, amplitude):
        """
        Construct the modulation (a triangular window) that will be used to modify the signal.

        The modulation is designed to affect the signal around a specific point of interest
        by applying a triangular window of specified width and amplitude.

        Args:
            data_axis (torch.Tensor): The axis representing the data.
            window_center (float): The point of interest.
            window_size (float): The width of the window around the point of interest.
            amplitude (float): The amplitude factor for the triangular window.

        Returns:
            torch.Tensor: The constructed modulation window.
        """
        window = torch.ones_like(self.data_axis)  # Initialize a window with ones
        mod_len = torch.sum(torch.abs(data_axis - window_center) < window_size).item()  # Determine the length of the triangular window
        mod_window = 1 - torch.from_numpy(windows.triang(mod_len)) * amplitude  # Create the triangular window and scale it by amplitude

        # Apply the window only to the region close to the point of interest
        mask = torch.abs(data_axis - window_center) < window_size
        window[mask] = mod_window.to(window.dtype)  # Ensure the dtype of mod_window matches that of window
        return window

    def apply_modulation(self, signal, modulation):
        """
        Modify the signal by applying the modulation window.

        This method scales the signal logarithmically before applying the modulation window to simulate
        the addition or attenuation of harmonics in the data.

        Args:
            signal (torch.Tensor): The original 1D signal.
            modulation (torch.Tensor): The modulation window to be applied.

        Returns:
            torch.Tensor: The modified signal.
        """
        min_signal = torch.min(signal)  # Find the minimum value in the signal
        max_signal = torch.max(signal)  # Find the maximum value in the signal
        signal = (signal - min_signal) / (max_signal - min_signal)  # Normalize the signal to the range [0, 1]

        mod_signal = signal + (1 - modulation).t()  # Apply the modulation window

        # Rescale the modified signal back to its original range
        mod_signal = mod_signal * (max_signal - min_signal) + min_signal
        return mod_signal.t()

    def forward(self, signal):
        """
        Forward pass to apply the harmonic modification to the signal.

        Args:
            signal (torch.Tensor): The original 1D signal.

        Returns:
            torch.Tensor: The modified signal after applying the modulation window.
        """
        mod_signal = self.apply_modulation(signal, self.modulation)
        return mod_signal
