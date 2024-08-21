import torch
from torch import nn
import matplotlib.pyplot as plt

class DelayPart(nn.Module):
    """
    A module to apply a delay to a specific windowed part of a 1D signal.

    This class is designed to shift a windowed segment of a signal by a specified delay,
    with an option to remove artifacts through interpolation. It can be used to simulate
    time shifts or delays in signals, particularly useful for testing anomaly detection algorithms.
    
    Attributes:
        data_axis (torch.Tensor): The axis representing the data (e.g., frequency or time).
        window_center_idx (int): The index corresponding to the center of the window in the data.
        delay_idx (int): The index corresponding to the delay applied to the windowed part of the signal.
        window_size_idx (int): The index corresponding to the size of the window around the center.
        remove_artifacts (bool): Flag to indicate whether to remove artifacts by interpolation.
        window_start (int): The start index of the window.
        window_end (int): The end index of the window.
        shifted_window_start (int): The start index of the shifted window.
        shifted_window_end (int): The end index of the shifted window.
        window_shifted_idx (torch.Tensor): Indices of the shifted window.
        window_idx (torch.Tensor): Indices of the original window.
        unions_idx (torch.Tensor): Unique indices from the union of original and shifted windows.
    """

    def __init__(self, data_axis, window_center, delay, window_size, remove_artifacts=False):
        """
        Initialize the DelayPart module.

        Args:
            data_axis (array-like): The axis representing the data (e.g., frequency or time).
            window_center (float): The center of the window in the data axis.
            delay (float): The delay to apply to the windowed part of the signal.
            window_size (float): The size of the window around the center.
            remove_artifacts (bool): Flag to indicate whether to remove artifacts by interpolation.
        """
        super().__init__()
        self.data_axis = data_axis
        self.window_center_idx = int(window_center / data_axis[1])
        self.delay_idx = int(delay / data_axis[1])
        self.window_size_idx = int(window_size / data_axis[1])
        self.remove_artifacts = remove_artifacts
        self.define_window_boundaries()
        self.calculate_shifted_boundary()
        self.indices_of_windows()

    def define_window_boundaries(self):
        """
        Define the start and end boundaries of the window based on the center and size.
        """
        self.window_start = max(0, self.window_center_idx - self.window_size_idx)
        self.window_end = min(len(self.data_axis), self.window_center_idx + self.window_size_idx)
        
    def calculate_shifted_boundary(self):
        """
        Calculate the boundaries of the window after applying the delay.
        """
        self.shifted_window_start = max(0, self.window_start - self.delay_idx)
        self.shifted_window_end = min(self.shifted_window_start + (self.window_end - self.window_start), len(self.data_axis))
    
    def indices_of_windows(self):
        """
        Generate the indices for the original and shifted windows, and their union.
        """
        self.window_shifted_idx = torch.tensor(range(self.shifted_window_start, self.shifted_window_end))
        self.window_idx = torch.tensor(range(self.window_start, self.window_end))
        self.unions_idx = torch.cat((self.window_shifted_idx, self.window_idx)).unique()

    def forward(self, signal):
        """
        Apply the delay to the windowed part of the signal.

        Args:
            signal (torch.Tensor): The original 1D signal.

        Returns:
            torch.Tensor: The modified signal with the windowed part delayed.
        """
        res = signal.clone()

        # Shift the windowed part of the signal
        res[self.window_shifted_idx] = signal[self.window_idx]

        # Subtract the interpolated linear difference to remove artifacts
        if self.remove_artifacts:
            res[self.window_shifted_idx] -= torch.linspace(
                res[self.window_shifted_idx[0]] - signal[self.window_shifted_idx[0]],
                res[self.window_shifted_idx[-1]] - signal[self.window_shifted_idx[-1]],
                len(self.window_shifted_idx)
            )

        # Handle non-overlapping parts of the original and shifted windows
        non_intersection = list(set(self.unions_idx.numpy()) - set(self.window_shifted_idx.numpy()))
        non_intersection.sort()

        if self.delay_idx > 0:
            res[non_intersection] = torch.linspace(
                res[non_intersection[0] - 1],
                signal[non_intersection[-1]],
                len(non_intersection)
            )
        elif self.delay_idx < 0:
            res[non_intersection] = torch.linspace(
                signal[non_intersection[0]],
                res[non_intersection[-1] + 1],
                len(non_intersection)
            )
            
        return res


