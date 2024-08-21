import unittest
import torch
from scipy import signal
from virtual_anomaly import AddSpike  # Replace 'your_module' with the actual module name
import matplotlib.pyplot as plt

class TestAddSpike(unittest.TestCase):
    def setUp(self):
        """
        Set up test variables before each test.
        """
        self.data_axis = torch.arange(0, 100, 1/25)  # Simulated data axis
        self.signal = (
            torch.sin(2 * torch.pi * self.data_axis) +
            0.5 * torch.sin(4 * torch.pi * self.data_axis) +
            0.9 * torch.sin(6 * torch.pi * self.data_axis) + 
            0.1 * torch.randn_like(self.data_axis)  # Add noise to the signal
        )
        
        self.window_center = 0.5
        self.window_size = 0.1
        self.amplitude = 0.5
        self.add_spike = AddSpike(
            data_axis=self.data_axis,
            window_center=self.window_center,
            window_size=self.window_size,
            amplitude=self.amplitude
        )

    def test_modulation_construction(self):
        """
        Test if the modulation window is constructed correctly.
        """
        modulation = self.add_spike.construct_modulation(
            self.data_axis,
            self.window_center,
            self.window_size,
            self.amplitude
        )

        self.assertEqual(modulation.shape, self.data_axis.shape)
        self.assertTrue(torch.all(modulation >= 0), "Modulation window should not have negative values.")
        self.assertTrue(torch.all(modulation <= 1), "Modulation window should not exceed 1.")


    def test_forward(self):
        """
        Test the forward method to ensure it processes the signal correctly.
        """
        output_signal = self.add_spike(self.signal)

        # Ensure the output shape is correct
        self.assertEqual(output_signal.shape, self.signal.shape)

        # Ensure that the output signal is indeed modified
        self.assertFalse(torch.equal(self.signal, output_signal), "The output signal should differ from the input signal after applying the forward method.")

    def test_validate_psd(self):
        f, Pxx = signal.welch(self.signal, fs=25)
        f, Pxx = torch.tensor(f), torch.tensor(Pxx)
        addspike = AddSpike(data_axis=f, window_center=f[50], window_size=f[2], amplitude=0.5)
        addspike2 = AddSpike(data_axis=f, window_center=f[60], window_size=f[2], amplitude=-0.3)

        output_signal = addspike(Pxx)
        output_signal2 = addspike2(Pxx)
        plt.plot(f, Pxx, label='Original Signal')
        plt.plot(f, output_signal, label='Affected Signal')
        plt.plot(f, output_signal2, label='Affected Signal 2')
        plt.legend()
        plt.savefig('tests/img/test_add_spike.png')

if __name__ == '__main__':
    unittest.main()