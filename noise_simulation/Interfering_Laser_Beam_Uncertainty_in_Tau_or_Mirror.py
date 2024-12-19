import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
wavelength1 = 633e-9  # Wavelength 1 in meters
wavelength2 = 639e-9  # Wavelength 2 in meters
c = 3e8  # Speed of light in m/s
f1 = c / wavelength1  # Frequency 1
f2 = c / wavelength2  # Frequency 2
T1 = 1 / f1  # Period 1
T2 = 1 / f2  # Period 2
w1 = 2 * math.pi * f1  # Angular frequency 1
w2 = 2 * math.pi * f2  # Angular frequency 2

# Beat frequency and period
f_beat = abs(f1 - f2)  # Beat frequency (Hz)
T_beat = 1 / f_beat  # Beat period (seconds)

# Nyquist frequency and time step calculation
nyquist = 2 * max(f1, f2)  # Nyquist rate based on highest frequency
step = 1 / nyquist  # Time step based on Nyquist criterion

# Useful print statements for understanding
print(f"Wavelength 1: {wavelength1 * 1e9} nm")
print(f"Wavelength 2: {wavelength2 * 1e9} nm")
print(f"Frequency 1: {f1:.2e} Hz")
print(f"Frequency 2: {f2:.2e} Hz")
print(f"Period of rapid oscillations for wave 1 (633 nm): {T1 * 1e15:.2f} fs")
print(f"Period of rapid oscillations for wave 2 (639 nm): {T2 * 1e15:.2f} fs")
print(f"Beat Frequency: {f_beat:.2e} Hz")
print(f"Nyquist Step Size: {step:.2e} seconds")
print(f"Beat Period: {T_beat * 1e15:.2f} fs")

# Number of points for high-resolution simulation
num_points = 100000

# Time array (focused on a small window, such as 200 fs)
t = np.linspace(0, step * num_points, num_points)  # Time window

# Array of time delays (varying tau, in femtoseconds)
# Extend tau beyond the beat period to observe full cycles of interference
tau_values = np.linspace(0, 5 * T_beat, 2000)  # Tau from 0 to 5 * T_beat

# Convert tau_values from seconds to femtoseconds for easier interpretation
tau_values_fs = tau_values * 1e15  # Tau in femtoseconds

# Introduce uncertainty in the mirror position (random fluctuations in tau)
mirror_uncertainty_std = 10e-18  # Standard deviation in seconds (10 attoseconds)
tau_uncertainty = np.random.normal(0, mirror_uncertainty_std, len(tau_values))  # Random Gaussian noise

# Apply uncertainty to tau values
tau_values_with_uncertainty = tau_values + tau_uncertainty

# To store average intensities for each tau
average_intensity_clean = np.zeros(len(tau_values))  # For clean signal
average_intensity_uncertain = np.zeros(len(tau_values))  # For signal with mirror uncertainty

# Loop over different tau values to compute the average intensity
for i, tau in enumerate(tau_values):
    # Create electric fields
    E1 = np.cos(w1 * t)  # Electric field 1
    E2 = np.cos(w2 * t)  # Electric field 2

    # Instead of np.roll, we incorporate tau directly in the cosine arguments
    E1_tau = np.cos(w1 * (t + tau))  # E1(t + tau)
    E2_tau = np.cos(w2 * (t + tau))  # E2(t + tau)

    # Combined electric field with time delay
    E_combined_tau = E1_tau + E2_tau

    # Compute the intensity |E_combined(t) + E_combined(t + tau)|^2 (clean signal)
    intensity_clean = np.abs(E1 + E2 + E_combined_tau) ** 2
    average_intensity_clean[i] = np.mean(intensity_clean)

# Repeat the same calculation for the signal with mirror position uncertainty
for i, tau in enumerate(tau_values_with_uncertainty):
    # Create electric fields
    E1 = np.cos(w1 * t)  # Electric field 1
    E2 = np.cos(w2 * t)  # Electric field 2

    # Instead of np.roll, we incorporate tau directly in the cosine arguments
    E1_tau = np.cos(w1 * (t + tau))  # E1(t + tau)
    E2_tau = np.cos(w2 * (t + tau))  # E2(t + tau)

    # Combined electric field with time delay
    E_combined_tau = E1_tau + E2_tau

    # Compute the intensity with mirror uncertainty
    intensity_uncertain = np.abs(E1 + E2 + E_combined_tau) ** 2
    average_intensity_uncertain[i] = np.mean(intensity_uncertain)

# Plotting the clean signal vs the signal with mirror uncertainty
plt.figure(figsize=(14, 8))
plt.plot(tau_values_fs, average_intensity_clean, '-', label='Clean Signal (No Uncertainty)', color='blue', linewidth=2)
plt.plot(tau_values_fs, average_intensity_uncertain, '--', label='Signal with Mirror Uncertainty', color='red', linewidth=2, alpha=0.7)
plt.title('Clean Signal vs. Signal with Mirror Position Uncertainty')
plt.xlabel('Time Delay τ (fs)')
plt.ylabel('Average Intensity')
plt.legend()
plt.grid()

# Zoom into a specific region of the graph for better comparison
plt.xlim([tau_values_fs[0], tau_values_fs[500]])  # Zoom in on the first part of the graph
plt.show()

# Now let's compute the FFT of both the clean and uncertain signals
fft_clean_intensity = np.fft.fft(average_intensity_clean)
fft_uncertain_intensity = np.fft.fft(average_intensity_uncertain)
fft_freqs = np.fft.fftfreq(len(average_intensity_clean), tau_values[1] - tau_values[0])

# Only use the positive frequencies and avoid zero division
positive_freqs = fft_freqs[fft_freqs > 0]  # Filter positive frequencies
positive_fft_clean_intensity = np.abs(fft_clean_intensity[fft_freqs > 0])  # FFT of clean signal
positive_fft_uncertain_intensity = np.abs(fft_uncertain_intensity[fft_freqs > 0])  # FFT of uncertain signal

# Convert positive frequencies to wavelengths
fft_wavelengths = c / positive_freqs  # λ = c / f

# Plotting Wavelength vs FFT Intensity for both signals
plt.figure(figsize=(14, 8))
plt.plot(fft_wavelengths * 1e9, positive_fft_clean_intensity, label='Clean FFT Intensity', color='blue', linewidth=2)
plt.plot(fft_wavelengths * 1e9, positive_fft_uncertain_intensity, label='Uncertain FFT Intensity', color='red', linewidth=2, alpha=0.7)
plt.title('FFT Intensity vs. Wavelength (Clean vs. Mirror Uncertainty)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('FFT Intensity')
plt.xlim([600, 700])  # Focus on the range of relevant wavelengths (adjust as needed)
plt.legend()
plt.grid()
plt.show()
