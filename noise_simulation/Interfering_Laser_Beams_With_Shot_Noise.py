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

# Number of points for high-resolution simulation
num_points = 100000

# Time array (focused on a small window, such as 200 fs)
t = np.linspace(0, step * num_points, num_points)  # Time window

# Array of time delays (varying tau, in femtoseconds)
tau_values = np.linspace(0, 5 * T_beat, 2000)  # Tau from 0 to 5 * T_beat

# Convert tau_values from seconds to femtoseconds for easier interpretation
tau_values_fs = tau_values * 1e15  # Tau in femtoseconds

# To store true (clean) and noisy average intensities for each tau
true_average_intensity = np.zeros(len(tau_values))
noisy_average_intensity = np.zeros(len(tau_values))

# Scaling factor for photon counts (to simulate shot noise)
scaling_factor = 1000

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

    # Compute the true intensity (without noise)
    intensity = np.abs(E1 + E2 + E_combined_tau) ** 2
    true_average_intensity[i] = np.mean(intensity)  # Store clean intensity

    # Simulate photon counts by scaling up intensity
    scaled_intensity = intensity * scaling_factor
    
    #  Change how noise is added to the signal

    # Apply Poisson noise to simulate shot noise
    noisy_intensity = np.random.poisson(scaled_intensity)

    # Scale back down after applying Poisson noise
    noisy_intensity = noisy_intensity / scaling_factor

    # Store noisy average intensity
    noisy_average_intensity[i] = np.mean(noisy_intensity)

# Plotting the true signal vs the signal with shot noise
plt.figure(figsize=(14, 8))
plt.plot(tau_values_fs, true_average_intensity, '-', label='True Signal (Without Noise)', color='blue', linewidth=2)
plt.plot(tau_values_fs, noisy_average_intensity, '--', label='Signal with Shot Noise', color='red', linewidth=2, alpha=0.7)
plt.title('True Signal vs. Signal with Shot Noise')
plt.xlabel('Time Delay τ (fs)')
plt.ylabel('Average Intensity')
plt.legend()
plt.grid()

# Zoom into a specific region of the graph for better comparison
plt.xlim([tau_values_fs[0], tau_values_fs[500]])  # Zoom in on the first part of the graph
plt.show()

# Plotting the difference between the true signal and the signal with shot noise
difference = noisy_average_intensity - true_average_intensity

plt.figure(figsize=(14, 8))
plt.plot(tau_values_fs, difference, '-', label='Difference (Noisy - True)', color='green', linewidth=2)
plt.title('Difference Between Noisy and True Signal')
plt.xlabel('Time Delay τ (fs)')
plt.ylabel('Difference in Intensity')
plt.legend()
plt.grid()

# Zoom in to a specific region of the graph for better comparison
plt.xlim([tau_values_fs[0], tau_values_fs[500]])  # Zoom in on the first part of the graph
plt.show()
