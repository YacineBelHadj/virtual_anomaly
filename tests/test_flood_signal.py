import unittest
import torch
import matplotlib.pyplot as plt
from virtual_anomaly import FloodSignal  # Replace with actual module import
from scipy import signal
class TestFloodSignal(unittest.TestCase):
    def setUp(self):
        """
        Set up test variables before each test.
        """
        self.signal = torch.linspace(0, 10, 500)  # Simulated 1D signal
        self.noise_level = 0.1
        self.flood_signal = FloodSignal(noise_level=self.noise_level)

    def test_flooding_effect(self):
        """
        Test if the noise flooding modifies the signal as expected.
        """
        output_signal = self.flood_signal(self.signal)

        # Ensure the output shape is correct
        self.assertEqual(output_signal.shape, self.signal.shape)

        # Ensure the signal is modified
        self.assertFalse(torch.equal(self.signal, output_signal), "The signal should be modified after applying flooding.")

    def test_noise_above_signal(self):
        """
        Ensure that noise flooding behaves correctly when the signal is below the noise level.
        """
        signal_below_noise = torch.zeros_like(self.signal)
        flooded_signal = self.flood_signal(signal_below_noise)
        self.assertTrue(flooded_signal.shape == signal_below_noise.shape, "The shape of the flooded signal should match the input signal.")
        self.assertTrue(torch.all(flooded_signal >= signal_below_noise), "Flooded signal should be equal to or greater than the original signal.")

    def test_plot_flooding_effect(self):
        """
        Generate a signal with multiple sine waves and noise, then visualize the power spectral density (PSD) 
        of the original and flooded signal.
        """
        # Generate a signal composed of multiple sine waves with noise
        t = torch.linspace(0, 30, 500)
        signal_wave = 10* torch.sin(2 * torch.pi * 3 * t) + torch.sin(2 * torch.pi * 20 * t) + 3* torch.sin(2 * torch.pi * 30 * t)
        noise = 0.1 * torch.randn_like(t)
        noisy_signal = signal_wave + noise

        # Apply FloodSignal

        # Compute the Power Spectral Density using Welch's method
        f_original, Pxx_original = signal.welch(noisy_signal.numpy(), fs=500, nperseg=256)
        Pxx_original = torch.tensor(Pxx_original)
        Pxx_flooded = self.flood_signal(Pxx_original)
        
        # Plot the Power Spectral Density
        plt.figure(figsize=(10, 4))
        plt.semilogy(f_original, Pxx_original, label="Original Signal PSD")
        plt.semilogy(f_original, Pxx_flooded, label="Flooded Signal PSD", linestyle='--')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density')
        plt.legend()
        plt.title("Power Spectral Density of Original and Flooded Signal")
        plt.savefig("tests/img/floodsignal_psd_test_output.png")


if __name__ == '__main__':
    unittest.main()
