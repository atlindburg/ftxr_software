import importlib
import FTI_library_AL as fti
importlib.reload(fti)


"""
Function: test_enhanced_frequency_detection()
Purpose: Tests the detection of specific frequencies in a signal using FFT analysis
Steps:
1. Creates a test signal with known frequencies (10, 50, 100 Hz)
2. Adds controlled noise to simulate real conditions
3. Performs FFT to analyze frequency components
4. Calculates power spectrum to measure frequency strengths
5. Detects and returns peaks in the frequency spectrum
"""
def test_enhanced_frequency_detection():
    # Set up sampling parameters
    sample_rate = 2000  # Hz - Nyquist frequency will be 1000 Hz
    duration = 1.0      # Total sampling duration in seconds
    t = fti.np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate test signal with multiple frequency components
    freqs = [10, 50, 100]  # Target frequencies in Hz
    amps = [2.0, 1.0, 0.5]  # Amplitudes in nm
    test_signal = fti.np.zeros_like(t)
    for freq, amp in zip(freqs, amps):
        test_signal += amp * fti.np.sin(2 * fti.np.pi * freq * t)
    
    # Add Gaussian noise to simulate measurement noise
    noise_level = 0.1
    test_signal += noise_level * fti.np.random.normal(0, 1, len(t))
    
    # Compute FFT and corresponding frequencies
    fft_result = fti.np.fft.fft(test_signal)
    fft_freq = fti.np.fft.fftfreq(len(t), 1/sample_rate)
    
    # Calculate power spectrum for amplitude analysis
    power_spectrum = fti.np.abs(fft_result)**2 / len(t)
    
    # Detect peaks in first half of spectrum (positive frequencies only)
    peaks, _ = fti.find_peaks(power_spectrum[:len(t)//2], height=0.1)
    detected_freqs = fti.np.abs(fft_freq[peaks])
    
    return detected_freqs, power_spectrum[peaks]

"""
Function: test_surface_roughness_characteristics()
Purpose: Analyzes different types of surface patterns and calculates their roughness parameters
Steps:
1. Generates various surface patterns (smooth, rough, periodic, step)
2. Calculates standard roughness metrics for each pattern
3. Returns comprehensive surface characterization results
"""
def test_surface_roughness_characteristics():
    def generate_surface_pattern(pattern_type):
        t = fti.np.linspace(0, 1, 1000)
        # Generate different surface profiles for testing
        if pattern_type == 'smooth':
            return 0.1 * fti.np.sin(2 * fti.np.pi * 10 * t)  # Low amplitude sine wave
        elif pattern_type == 'rough':
            return fti.np.random.normal(0, 1, len(t))        # Random Gaussian surface
        elif pattern_type == 'periodic':
            # Combination of two frequencies
            return fti.np.sin(2 * fti.np.pi * 10 * t) + 0.5 * fti.np.sin(2 * fti.np.pi * 20 * t)
        elif pattern_type == 'step':
            return fti.np.concatenate([fti.np.zeros(500), fti.np.ones(500)])  # Step function
    
    # Process each surface pattern type
    patterns = ['smooth', 'rough', 'periodic', 'step']
    results = {}
    
    for pattern in patterns:
        surface = generate_surface_pattern(pattern)
        
        # Calculate standard roughness parameters
        rms = fti.np.sqrt(fti.np.mean(surface**2))      # Root Mean Square roughness
        ra = fti.np.mean(fti.np.abs(surface))           # Average roughness
        rmax = fti.np.max(fti.np.abs(surface))          # Maximum peak height
        skewness = fti.np.mean(surface**3) / (rms**3)   # Distribution asymmetry
        kurtosis = fti.np.mean(surface**4) / (rms**4)   # Peak sharpness
        
        results[pattern] = {
            'RMS': rms,
            'Ra': ra,
            'Rmax': rmax,
            'Skewness': skewness,
            'Kurtosis': kurtosis
        }
    
    return results

"""
Function: test_spatial_frequency_analysis()
Purpose: Performs 2D spatial frequency analysis on a generated surface
Steps:
1. Creates a 2D surface with known spatial frequencies
2. Performs 2D FFT analysis
3. Calculates power spectral density
4. Computes radially averaged PSD for isotropic analysis
"""
def test_spatial_frequency_analysis():
    # Generate 2D surface grid
    size = 100
    x = fti.np.linspace(0, 1, size)
    y = fti.np.linspace(0, 1, size)
    X, Y = fti.np.meshgrid(x, y)
    
    # Create test surface with multiple spatial components
    surface = (
        fti.np.sin(2 * fti.np.pi * 5 * X) +      # Low frequency X component
        0.5 * fti.np.sin(2 * fti.np.pi * 20 * Y) + # High frequency Y component
        0.2 * fti.np.random.normal(0, 1, (size, size))  # Random roughness
    )
    
    # Perform 2D FFT and shift zero frequency to center
    fft2d = fti.np.fft.fft2(surface)
    fft2d_shifted = fti.np.fft.fftshift(fft2d)
    
    # Calculate power spectral density
    psd2d = fti.np.abs(fft2d_shifted)**2
    
    # Compute radially averaged PSD
    center = size // 2
    r = fti.np.linspace(0, center, center)
    psd_radial = fti.np.zeros_like(r)
    
    # Average PSD over circular rings
    for i, radius in enumerate(r):
        mask = fti.np.logical_and(
            fti.np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) >= radius/center,
            fti.np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) < (radius+1)/center
        )
        psd_radial[i] = fti.np.mean(psd2d[mask])
    
    return psd_radial, r

def run_all_tests():
    """
    Runs all test functions and displays their results in a formatted way
    """
    print("\n=== Running All FFT Analysis Tests ===\n")
    
    # Run frequency detection test
    print("1. Frequency Detection Test")
    print("-" * 30)
    detected_freqs, peak_powers = test_enhanced_frequency_detection()
    for freq, power in zip(detected_freqs, peak_powers):
        print(f"Frequency: {freq:.1f} Hz, Power: {power:.2f}")
    
    # Run surface roughness tests
    print("\n2. Surface Roughness Test")
    print("-" * 30)
    roughness_results = test_surface_roughness_characteristics()
    for pattern, metrics in roughness_results.items():
        print(f"\n{pattern.capitalize()} surface:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
    
    # Run spatial frequency analysis
    print("\n3. Spatial Frequency Analysis")
    print("-" * 30)
    psd_radial, spatial_freqs = test_spatial_frequency_analysis()
    print("Analysis completed successfully")
    print("\n=== All Tests Completed ===\n")
    
    return {
        'frequency_detection': (detected_freqs, peak_powers),
        'roughness_analysis': roughness_results,
        'spatial_analysis': (psd_radial, spatial_freqs)
    }