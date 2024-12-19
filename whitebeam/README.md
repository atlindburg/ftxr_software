# White Beam Simulation and FFT Analysis

## Overview

This Python script simulates the interference of a white light beam (with wavelengths ranging from 400 nm to 700 nm) by calculating the average intensity over various time delays (τ). It also computes the Fourier Transform (FFT) of the intensity to analyze the frequency components of the beam and generates plots for visualization.

## Key Features

1. **Electric Field Simulation:**
   - The script models the electric fields of a white light beam over a range of wavelengths (400 nm to 700 nm).
   - It calculates the electric fields for each wavelength as a function of time and time delay (τ).
   - The fields are summed to compute the total electric field at each time delay.

2. **Time Delay and Intensity Calculation:**
   - For each time delay, the script computes the total electric field and calculates the intensity (proportional to the square of the electric field).
   - The average intensity over time is computed for different time delays and plotted as **Average Intensity vs. Time Delay (τ)**.

3. **FFT Analysis:**
   - The script applies a **Fourier Transform (FFT)** to the intensity data to analyze the frequency components of the white beam.
   - The resulting frequency components are converted into **wavelength** and plotted as **FFT Intensity vs. Wavelength**.
   - This helps visualize how different wavelengths contribute to the overall intensity of the white beam.

## Output

1. **Average Intensity vs. Time Delay (τ):**
   - This plot shows how the average intensity varies as a function of the time delay between the electric fields.

2. **FFT Intensity vs. Wavelength:**
   - This plot shows the Fourier Transform of the intensity, providing insight into how different wavelengths contribute to the overall intensity of the white beam.

Both plots are saved as `.png` files in the working directory:
- `average_intensity_vs_time_delay.png`
- `fft_intensity_vs_wavelength.png`

## How to Run the Script

1. Ensure you have Python installed with the following libraries:
   - `numpy`
   - `matplotlib`

2. Run the script in a Python environment. The graphs will be displayed and saved to the working directory as `.png` files.

## Code Highlights

- **Wavelengths:** Ranges from 400 nm to 700 nm.
- **Time Delays:** A fine time delay grid is used near τ = 0 to ensure accurate resolution of intensity peaks.
- **FFT:** The script computes the FFT to analyze the frequency components and convert them into wavelength space.
