import unittest
import torch
import matplotlib.pyplot as plt
from virtual_anomaly import DelayPart 

class TestDelayPart(unittest.TestCase):
    def setUp(self):
        """
        Set up test variables before each test.
        """
        self.data_axis = torch.linspace(0, 1, 500)  # Simulated data axis (e.g., frequency)
        # make a chainsaw-like signal 
        self.signal = torch.linspace(0, -5, 500)
        self.spike = torch.cat((torch.linspace(0, 1, 10), torch.linspace(1,0, 10)))
        self.signal[20:40] += self.spike
        self.signal[100:120] += self.spike
        self.signal[200:220] += self.spike
        self.window_center = 0.2
        self.delay = 0.05
        self.window_size = 0.05
        self.remove_artifacts = False
        
        self.delay_part = DelayPart(
            data_axis=self.data_axis,
            window_center=self.window_center,
            delay=self.delay,
            window_size=self.window_size,
            remove_artifacts=self.remove_artifacts
        )


    def test_window_boundaries(self):
        """
        Test if the window boundaries are calculated correctly.
        """
        self.delay_part.define_window_boundaries()
        
        expected_window_start = max(0, self.delay_part.window_center_idx - self.delay_part.window_size_idx)
        expected_window_end = min(len(self.data_axis), self.delay_part.window_center_idx + self.delay_part.window_size_idx)
        
        self.assertEqual(self.delay_part.window_start, expected_window_start)
        self.assertEqual(self.delay_part.window_end, expected_window_end)

    def test_shifted_boundary(self):
        """
        Test if the shifted window boundaries are calculated correctly.
        """
        self.delay_part.calculate_shifted_boundary()
        
        expected_shifted_start = max(0, self.delay_part.window_start - self.delay_part.delay_idx)
        expected_shifted_end = min(expected_shifted_start + (self.delay_part.window_end - self.delay_part.window_start), len(self.data_axis))
        
        self.assertEqual(self.delay_part.shifted_window_start, expected_shifted_start)
        self.assertEqual(self.delay_part.shifted_window_end, expected_shifted_end)

    def test_indices_of_windows(self):
        """
        Test if the indices of the original and shifted windows are calculated correctly.
        """
        self.delay_part.indices_of_windows()
        
        self.assertTrue(len(self.delay_part.window_shifted_idx) > 0, "Shifted window indices should not be empty.")
        self.assertTrue(len(self.delay_part.window_idx) > 0, "Original window indices should not be empty.")
        self.assertTrue(len(self.delay_part.unions_idx) > 0, "Union of window indices should not be empty.")

    def test_forward(self):
        """
        Test the forward method to ensure it processes the signal correctly.
        """
        output_signal = self.delay_part(self.signal)

        # Ensure the output shape is correct
        self.assertEqual(output_signal.shape, self.signal.shape)

        # Ensure that the output signal is indeed modified
        self.assertFalse(torch.equal(self.signal, output_signal), "The output signal should differ from the input signal after applying the forward method.")

    def test_validate_output_plot(self):
        """
        Validate the output by generating and saving a plot.
        """
        output_signal = self.delay_part(self.signal)
        delay_part_rm_artifacts = DelayPart(
            data_axis=self.data_axis,
            window_center=self.window_center,
            delay=self.delay,
            window_size=self.window_size,
            remove_artifacts=True
        )
        output_signal_rm_artifacts = delay_part_rm_artifacts(self.signal)        
        plt.figure(figsize=(10, 4))
        plt.plot(self.data_axis, self.signal, label="Original Signal",lw=2)
        plt.plot(self.data_axis, output_signal, label="Modified Signal (Shifted and Windowed)",ls='--')
        plt.plot(self.data_axis, output_signal_rm_artifacts, label="Modified Signal (Shifted, Windowed, and Artifacts Removed)",ls='-.')
        plt.legend()
        plt.title("DelayPart Test Output")
        plt.savefig("tests/img/delaypart_test_output.png")

if __name__ == '__main__':
    unittest.main()
