# Virtual Anomaly Simulation 

## Overview

**Virtual Anomaly Simulation** is a Python package designed with torch to facilitate the testing of anomaly detection algorithms. Primarily focused on altering the spectral content of signals, this package can also be applied to any 1D data. It provides tools to modify and manipulate data, allowing you to simulate various anomalies and shifts, which are critical for robust algorithm testing.

## Features

- **DelayPart**: A module that shifts a windowed portion of a signal. It is particularly useful for testing the detection of delays or shifts in spectral content.
  
- **AddSpike**: A module that adds or alters harmonics within a PSD/ FFT, making it easier to test the ability of algorithms to detect changes in frequency components.

- **FloodSignal**: A module that Flood the low part of signal with noise

## Installation

You can install the package via pip:

```bash
pip install virual-anomaly
```
