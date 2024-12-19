import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
wavelength1 = 633e-9  # Wavelength 1 in meters
wavelength2 = 639e-9  # Wavelength 2 in meters
c = 3e8  # Speed of light in m/s
f1 = c / wavelength1  # Frequency 1
f2 = c / wavelength2  # Frequency 2
T1 = 1 / f1 # Period 1
T2 = 1 / f2 # Period 2
w1 = 2 * math.pi * f1  # Angular frequency 1
w2 = 2 * math.pi * f2  # Angular frequency 

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
tau_values = np.linspace(0, 5 * T_beat, 2000)  # Tau from 0 to 1.2 * T_beat

# Convert tau_values from seconds to femtoseconds for easier interpretation
tau_values_fs = tau_values * 1e15  # Tau in femtoseconds

# To store average intensities for each tau
average_intensity = np.zeros(len(tau_values))

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

    # Compute the intensity |E_combined(t) + E_combined(t + tau)|^2
    intensity = np.abs(E1 + E2 + E_combined_tau) ** 2

    # Compute the average intensity (mean over the entire time window)
    average_intensity[i] = np.mean(intensity)

# Plotting the average intensity vs tau
plt.figure(figsize=(10, 6))
plt.plot(tau_values_fs, average_intensity, '-o')  # Tau in femtoseconds
plt.title('Average Intensity vs. Time Delay (τ)')
plt.xlabel('Time Delay τ (fs)')
plt.ylabel('Average Intensity')
plt.grid()
plt.show()

# Now let's compute the FFT of the average intensity to get wavelength vs intensity
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
plt.title('FFT Intensity vs. Wavelength')
plt.xlabel('Wavelength (nm)')
plt.ylabel('FFT Intensity')
plt.xlim([600, 700])  # Focus on the range of relevant wavelengths (adjust as needed)
plt.grid()
plt.show()
