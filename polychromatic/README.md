# Beat Frequency and Interference Simulation for Two Wavelengths

## Overview
This Python script simulates the interference of two light beams with slightly different wavelengths (633 nm and 639 nm). It computes the average intensity of the combined electric fields as a function of time delay between them. Additionally, the script performs a Fast Fourier Transform (FFT) on the computed intensities to analyze the resulting beat frequency and wavelength spectrum. The results are visualized through two plots: one showing the average intensity vs. time delay and another showing the FFT intensity vs. wavelength.

## Analysis
In the Intensity Vs. Time Delay plot we notice the envelope of the wave goes to zero when 

## Requirements
To run the script, you need to have the following Python libraries installed:
- `numpy`
- `matplotlib`
- `math`

You can install the required libraries using:
```bash
pip install numpy matplotlib

