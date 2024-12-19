import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
wavelength_mono = 633e-9  # Monochromatic wavelength (633 nm) in meters
c = 3e8  # Speed of light in m/s
f_mono = c / wavelength_mono  # Frequency for monochromatic beam
w_mono = 2 * math.pi * f_mono  # Angular frequency for monochromatic beam

# Nyquist frequency and time step calculation
nyquist = 2 * f_mono  # Nyquist rate based on monochromatic frequency
step = 1 / nyquist  # Time step based on Nyquist criterion

# Number of points for high-resolution simulation
num_points = 10000

# Time array (for about one wavelength duration or more)
t = np.linspace(0, step * num_points, num_points)  # Time window

# Extended range of time delays (varying tau, in femtoseconds)
tau_values = np.linspace(0, 20 / f_mono, 2000)  # Extend tau to 20 periods for better resolution

# Convert tau_values from seconds to femtoseconds for easier interpretation
tau_values_fs = tau_values * 1e15  # Tau in femtoseconds

# To store average intensities for each tau
average_intensity = np.zeros(len(tau_values))

# Loop over different tau values to compute the average intensity
for i, tau in enumerate(tau_values):
    # Electric field for the monochromatic beam
    E_mono = np.cos(w_mono * t)  # Electric field with no delay
    E_mono_tau = np.cos(w_mono * (t + tau))  # Electric field with time delay

    # Compute the intensity |E(t) + E(t + tau)|^2
    intensity = np.abs(E_mono + E_mono_tau) ** 2

    # Compute the average intensity (mean over the entire time window)
    average_intensity[i] = np.mean(intensity)

# Plotting the average intensity vs tau
plt.figure(figsize=(10, 6))
plt.plot(tau_values_fs, average_intensity, '-o')  # Tau in femtoseconds
plt.title('Average Intensity vs. Time Delay (τ) for Monochromatic Beam')
plt.xlabel('Time Delay τ (fs)')
plt.ylabel('Average Intensity')
plt.grid()
plt.show()

# Compute the FFT of the average intensity
fft_intensity = np.fft.fft(average_intensity)
fft_freqs = np.fft.fftfreq(len(average_intensity), tau_values[1] - tau_values[0])

# Only use the positive frequencies and avoid zero division
positive_freqs = fft_freqs[fft_freqs > 0]  # Filter positive frequencies
positive_fft_intensity = np.abs(fft_intensity[fft_freqs > 0])  # Corresponding intensities

# Convert positive frequencies to wavelengths
fft_wavelengths = c / positive_freqs  # λ = c / f

# Plotting Wavelength vs FFT Intensity
plt.figure(figsize=(10, 6))
plt.plot(fft_wavelengths * 1e9, positive_fft_intensity)  # Convert wavelengths to nm
plt.title('FFT Intensity vs. Wavelength for Monochromatic Beam')
plt.xlabel('Wavelength (nm)')
plt.ylabel('FFT Intensity')
plt.xlim([600, 700])  # Focus on the range of relevant wavelengths (adjust as needed)
plt.grid()
plt.show()
