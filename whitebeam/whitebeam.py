import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
c = 3e8  # Speed of light in m/s

# Wavelength range for a white beam (visible spectrum ~400 nm to 700 nm)
wavelengths = np.linspace(400e-9, 700e-9, 100)  # 100 wavelengths from 400 nm to 700 nm

# Calculate frequencies and angular frequencies for each wavelength
frequencies = c / wavelengths  # Frequencies corresponding to the wavelengths
angular_frequencies = 2 * math.pi * frequencies  # Angular frequencies

# Print the range of wavelengths and frequencies
print(f"Wavelength range (nm): {wavelengths[0]*1e9:.2f} nm to {wavelengths[-1]*1e9:.2f} nm")
print(f"Frequency range (THz): {frequencies[0]*1e-12:.2f} THz to {frequencies[-1]*1e-12:.2f} THz")

# Nyquist frequency and time step calculation
nyquist = 2 * max(frequencies)  # Nyquist rate based on highest frequency
step = 1 / nyquist  # Time step based on Nyquist criterion

# Print Nyquist frequency and time step
print(f"Nyquist frequency (Hz): {nyquist:.2e} Hz")
print(f"Time step (s): {step:.2e} s")

# Number of points for high-resolution simulation
num_points = 100000

# Time array is unnecessary for this simulation
# Time array (focused on a small window, such as 200 fs)
t = np.linspace(0, step * num_points, num_points)  # Time window

# Print sample electric fields for key wavelengths
print(f"Electric field at 400 nm (sample): {np.cos(angular_frequencies[0] * t)[:5]}")
print(f"Electric field at 700 nm (sample): {np.cos(angular_frequencies[-1] * t)[:5]}")

# Array of time delays (varying tau, in femtoseconds)
tau_values = np.linspace(0, 10e-15, 2500)  # Evenly spaced values from 0 to 10 fs
    np.linspace(10e-15, 200e-15, 5000)
tau_values_fs = tau_values * 1e15  # Convert tau_values to femtoseconds

# To store average intensities for each tau
average_intensity = np.zeros(len(tau_values))

# Loop over different tau values to compute the average intensity
# for i, tau in enumerate(tau_values):
#     E_combined = np.zeros(num_points)  # Initialize combined electric field
#     E_combined_tau = np.zeros(num_points)  # For time delay

#     # Loop through each wavelength
#     for w in angular_frequencies:
#         E = np.cos(w * t)  # Electric field for this wavelength
#         E_tau = np.cos(w * (t + tau))  # Electric field with time delay

#         E_combined += E / np.sqrt(len(wavelengths))
#         E_combined_tau += E_tau / np.sqrt(len(wavelengths))

#     # Compute the intensity |E(t) + E(t + tau)|^2
#     intensity = np.abs(E_combined + E_combined_tau) ** 2
#     average_intensity[i] = np.mean(intensity)

#     # Print key time delays and average intensity
#     if i % 500 == 0:
#         print(f"At tau = {tau_values_fs[i]:.2f} fs, average intensity = {average_intensity[i]:.4f}")


def I_tau(tau):
    Es = 0
    for w in angular_frequencies:
        Es = Es+ np.cos(w * (tau))
    return np.abs(Es)**2

for i_tau in range(len(tau_values)):
    average_intensity = I_tau(tau_values_fs)

# Plotting the average intensity vs tau
plt.figure(figsize=(10, 6))
plt.plot(tau_values_fs, average_intensity, '-o')  # Tau in femtoseconds
plt.title('Average Intensity vs. Time Delay (τ) for White Beam')
plt.xlabel('Time Delay τ (fs)')
plt.ylabel('Average Intensity')
plt.grid()
plt.savefig('average_intensity_vs_time_delay.png', dpi=300)  # Save the figure
plt.show()

# FFT of average intensity
fft_intensity = np.fft.fft(average_intensity)
fft_freqs = np.fft.fftfreq(len(average_intensity), tau_values[1] - tau_values[0])
positive_freqs = fft_freqs[fft_freqs > 0]
positive_fft_intensity = np.abs(fft_intensity[fft_freqs > 0])

# Print sample FFT frequencies and intensities
print("Sample FFT frequencies (THz):", positive_freqs[:5] * 1e-12)
print("Sample FFT intensities:", positive_fft_intensity[:5])

# Convert frequencies to wavelengths and plot
fft_wavelengths = c / positive_freqs
plt.figure(figsize=(10, 6))
plt.plot(fft_wavelengths * 1e9, positive_fft_intensity)  # Convert wavelengths to nm
plt.title('FFT Intensity vs. Wavelength for White Beam')
plt.xlabel('Wavelength (nm)')
plt.ylabel('FFT Intensity')
plt.xlim([400, 700])
plt.grid()
plt.savefig('fft_intensity_vs_wavelength.png', dpi=300)  # Save the figure
plt.show()

# Print sample FFT wavelengths and corresponding intensities
print("Sample FFT wavelengths (nm):", fft_wavelengths[:5] * 1e9)
print("Sample FFT intensities:", positive_fft_intensity[:5])