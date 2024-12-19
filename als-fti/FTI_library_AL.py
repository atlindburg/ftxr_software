import numpy as np
import matplotlib.pyplot as plt
import pypylon.pylon as pylon
from pipython import pitools, GCSDevice
import os as os
import time
from datetime import datetime
import unittest
from scipy.signal import find_peaks

# CONSTANTS

# Max velocity of pidevice
max_velocity = 1000000
# Speed of light
c = 3e8
# Current directory
current_directory = os.getcwd()

# -------------------------------------------------

# FUNCTIONS

def main_program():
    collect_images_fast(displacements_um, cam, pidevice, velocity)
    fft_images(displacements_um, center_x, center_y, img_x, img_y, current_directory, img_directory, output_file)
    analyze_FFT_data(filename)

def start_tools():
    """
    Initialize and configure measurement tools including GSCDevice and camera.
    
    This function performs the following steps:
    1. Checks for existing GSCDevice connection
    2. Initializes new GSCDevice if needed
    3. Sets up PI tools for the device
    4. Initializes and configures the camera
    5. Sets exposure time and displays initial image
    
    Returns:
        tuple: (pidevice, cam)
            - pidevice: Initialized and connected GSCDevice object
            - cam: Configured camera object
    
    Raises:
        Exception: If device initialization or camera setup fails
    """
    try:
        # Check if pidevice exists in global namespace and is connected
        if 'pidevice' in globals() and pidevice.IsConnected():
            print("pidevice already exists and is connected.")
        else:
            print("Initializing new GSCDevice...")
            pidevice = initialize_GSCDevice()
            
        # Verify connection status after initialization
        if pidevice.IsConnected():
            print("GSCDevice initialized and connected successfully.")
        else:
            print("GSCDevice initialized but not connected.")
            
    except NameError:
        # Handle case where pidevice variable doesn't exist
        print("pidevice not defined. Initializing new GSCDevice...")
        pidevice = initialize_GSCDevice()
    except Exception as e:
        # Catch and report any other initialization errors
        print(f"An error occurred: {str(e)}")
    
    # Initialize PI tools with connected device
    initialize_pitools(pidevice)
    
    # Set up camera and capture initial image
    cam, res = initialize_cam()
    
    # Configure camera exposure (5000 μs = 5 ms)
    new_exposure_time = set_exposure_time(cam, 5000)
    exposure_time = get_exposure_time(cam)
    
    # Display initial camera image
    show_picture(res)
    
    return pidevice, cam

def analyze_FFT_data(filename):
    displacements_um, pixel_sum, fft_result, fft_freq_Hz = load_FFT_data(filename)

def load_FFT_data(filename):
    # Load data from filename
    loaded_data = np.load(filename)
    # Store displacements data in an array
    displacements_um = loaded_data['displacements']
    # Store intensity data in an array
    pixel_sum = loaded_data['intensity']
    # Load FFT data
    fft_result = loaded_data['fft']
    # Load FFT freq
    fft_freq_Hz = loaded_data['fft_freq']
    # Return all data
    return displacements_um, pixel_sum, fft_result, fft_freq_Hz

def analyze_phase_data(fft_data, wavelength):
    """
    Analyze phase information from FFT data to reconstruct surface height.
    
    Args:
        fft_data: Structured array from fft_images containing FFT results
        wavelength: Laser wavelength in micrometers
    
    Returns:
        tuple: (heights, positions) Arrays of calculated heights and corresponding positions
    """
    # Extract phase from complex FFT data
    phases = np.angle(fft_data['fourier_transform'])
    
    # Unwrap phase along each pixel's frequency axis
    unwrapped_phases = np.unwrap(phases, axis=1)
    
    # Convert phase to height (λ/4π relationship)
    heights = (wavelength / (4 * np.pi)) * unwrapped_phases
    
    # Get corresponding pixel positions
    positions = fft_data['position']
    
    return heights, positions

def calculate_psd(fft_data):
    # Calculate Power Spectral Density
    return np.abs(fft_data)**2

def calculate_rms_roughness(psd_data, spatial_frequencies):
    # Calculate RMS roughness from PSD
    return np.sqrt(np.trapz(psd_data, spatial_frequencies))

def analyze_surface_roughness(fft_file):
    data = np.load(fft_file)
    
    for pixel in data:
        position, fft, frequencies = pixel
        psd = calculate_psd(fft)
        rms_roughness = calculate_rms_roughness(psd, frequencies)
        
        # Further analysis can be performed here
        
    # Combine results for all pixels and generate overall surface characterization

def collect_images_fast(displacements_um, cam, pidevice, velocity):

    """
    Function: collect_images_fast
    Purpose: Collects and saves images at different displacement positions using a camera and piezo device
    Parameters:
        displacements_um: Array of displacement positions in micrometers
        cam: Camera object for image capture
        pidevice: Piezo device controller
        velocity: Movement velocity for the piezo device
    Returns:
        images: List of captured images stored in numpy array
        run_dir: Directory path where images are saved in numpy arrays
    """

    # Set the velocity of the mirror mounting system
    pidevice.VEL('1', velocity)
    # Initialize list to store captured images
    images = []
    # Collect a times stamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a directory name using the timestamp
    run_dir = f"run_{timestamp}"
    # Create the directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Iterate through each displacement position
    # Pos is displacement step and i_d is index
    for i_d, pos in enumerate(displacements_um):
        # Move piezo to new position
        pidevice.MOV('1', pos)
        # Capture image with 100ms timeout
        res = cam.GrabOne(100)  # Reduced timeout
        # Convert image to numpy array
        img = np.array(res.Array)
        # Release camera buffer
        res.Release()  # Free up resources
        # Store image in memory
        images.append(img)
        # Save image to disk with formatted index
        np.save(os.path.join(run_dir, f"image_{i_d:04d}.npy"), img)
    
    # Save displacement positions for reference
    np.save(os.path.join(run_dir, "displacements.npy"), displacements_um)
    
    return images, run_dir

def collect_intensity_values_of_pixel(displacements_um, pixel_x_id, pixel_y_id, base_dir, folder):

    """
    Collects intensity values for a specific pixel across multiple displacement images.
    
    Args:
        displacements_um: Array of displacement values in micrometers
        pixel_x_id: X coordinate of the pixel to analyze
        pixel_y_id: Y coordinate of the pixel to analyze
        base_dir: Base directory containing image data
        folder: Specific folder name containing the image sequence
    
    Returns:
        data: Array of intensity values for the specified pixel
        image_shape: Shape of the image array (height, width)
    """

    # Figure out directory where image data is stored
    image_dir = os.path.join(base_dir, folder)
    # Create data array that is the size of the displacements array
    data = np.zeros_like(displacements_um)
    # Set image_shape to none
    image_shape = None

    # Loop through all the indexes of displacement step array
    for i_d in range(len(displacements_um)):
        # Establish path to image file we want to examine
        img_path = os.path.join(image_dir, f"image_{i_d:04d}.npy")
        # Check if image file exists
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue
        # Attempt to load image data from file 
        try:
            # Load image data from file
            img = np.load(img_path)
            # Collect intensity value in specific pixel
            data[i_d] = img[pixel_x_id, pixel_y_id]
            # Collect image shape.. not sure of purpose
            if image_shape is None:
                image_shape = img.shape
        # Throw an error if image cannot be loaded
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
    
    return data, image_shape

def fft_images(displacements_um, center_x, center_y, img_x, img_y, current_directory, img_directory, output_file):
    """
    Performs FFT analysis on a region of pixels from saved interferometric image data and computes wavelength data.
    
    Args:
        displacements_um (ndarray): Array of displacement values in micrometers
        center_x (int): X coordinate of region center
        center_y (int): Y coordinate of region center
        img_x (int): Width of region to analyze
        img_y (int): Height of region to analyze
        current_directory (str): Base directory path
        img_directory (str): Directory containing interferogram images
        output_file (str): Path to save FFT output data
        
    Returns:
        tuple: Shape of the original images (height, width)
        
    The function:
    1. Creates a structured array to store per-pixel data:
       - (x,y) pixel coordinates
       - Complex FFT results 
       - Corresponding frequency values
       - Wavelength data for positive frequencies
       - Magnitude spectrum for positive frequencies
       
    2. For each pixel in the specified region:
       - Loads intensity values across all displacement steps
       - Computes FFT of intensity vs displacement data
       - Computes wavelengths and magnitude spectrum
       - Stores all results in structured array
       
    3. Saves complete FFT dataset to .npy file
    
    Notes:
        - Region size should be chosen based on surface correlation length
        - Typically 10x10 to 32x32 pixels is sufficient
        - Requires collect_intensity_values_of_pixel(), compute_FFT_pixel(), and compute_wavelengths()
        - Memory usage scales with region size and number of displacement steps
        - Wavelength data is computed only for positive frequencies
    """

    # Create a structured array to hold the data
    dtype = [
        ('position', '2int32'),
        ('fourier_transform', 'complex128', (len(displacements_um),)),
        ('FFT_freq_Hz', 'float64', (len(displacements_um),)),
        ('wavelengths_m', 'float64', (len(displacements_um)//2 + 1)),  # Only positive frequencies
        ('magnitude_spectrum', 'float64', (len(displacements_um)//2 + 1))
    ]
    # Create a data array of dtype
    data = np.zeros(img_x * img_y, dtype=dtype)
    # Initialize index to 0
    index = 0

    # Loop through all pixels in a specifc area
    for j in range(center_x - img_x // 2, center_x + img_x//2):
        for k in range(center_y - img_y //2, center_y + img_y // 2):
            # Collect intensity values for a specific pixel across all displacement steps
            img_data, img_shape = collect_intensity_values_of_pixel(displacements_um, j, k, current_directory, img_directory)
            # Perform FFT for the pixel
            fourier_transform, FFT_freq_Hz = compute_FFT_pixel(displacements_um, img_data)
            # Get positive frequencies and corresponding wavelengths
            wavelengths_m, positive_magnitude = compute_wavelengths(fourier_transform, FFT_freq_Hz)
            # Perform padding to match expected size
            padded_wavelengths_m = np.pad(wavelengths_m, (0, (len(displacements_um) // 2 + 1 ) - len(wavelengths_m)), mode='constant')
            padded_positive_magnitude = np.pad(positive_magnitude, (0, (len(displacements_um) // 2 + 1 ) - len(wavelengths_m)), mode='constant')
            # Store the resulting FFT and FFT frequencies to data array documenting what pixel is analyzed
            data[index] = ((j, k), fourier_transform, FFT_freq_Hz, padded_wavelengths_m, padded_positive_magnitude)
            # Increment index
            index += 1

    # Save the FFT data to a .npy file
    np.save(output_file, data)

    return img_shape

def compute_wavelengths(fourier_transform, FFT_freq_Hz):
    """
    Convert FFT frequency data to wavelengths and compute magnitude spectrum.
    
    Args:
        fourier_transform (ndarray): Complex FFT results from signal analysis
        FFT_freq_Hz (ndarray): Frequency values in Hertz corresponding to FFT results
        
    Returns:
        tuple: (wavelengths_m, positive_magnitude)
            - wavelengths_m: Array of wavelengths in meters
            - positive_magnitude: Magnitude spectrum for positive frequencies
            
    Notes:
        - Only processes positive frequencies since FFT is symmetric
        - Uses speed of light (c) to convert frequencies to wavelengths
        - Magnitude spectrum computed as absolute value of complex FFT
        
    The function:
    1. Computes magnitude spectrum from complex FFT data
    2. Filters for positive frequencies only
    3. Converts frequencies to wavelengths using λ = c/f
    """

    # Compute magnitude spectrum
    magnitude = np.abs(fourier_transform)
    # Select only positive frequencies and their magnitudes
    positive_frequencies_Hz = FFT_freq_Hz[FFT_freq_Hz > 0]
    positive_magnitude = magnitude[FFT_freq_Hz > 0]
    # Convert frequencies to wavelengths using c (speed of light)
    wavelengths_m = c/positive_frequencies_Hz
    
    return wavelengths_m, positive_magnitude


def compute_FFT_pixel(displacements_um, pixel):
    """
    Compute the Fast Fourier Transform (FFT) for a single pixel's intensity values.
    
    Args:
        displacements_um (ndarray): Array of displacement values in micrometers
        pixel (ndarray): Array of intensity values for a single pixel
        
    Returns:
        tuple: (fourier_transform, FFT_freq)
            - fourier_transform: Complex FFT results
            - FFT_freq: Corresponding frequency values in Hz
    """
    # Perform FFT of pixel
    fourier_transform = np.fft.fft(pixel)
    # Find sampling period
    period_s = (2 * (displacements_um[1] - displacements_um[0])* 1e-6) / c
    # Store FFT frequencies
    FFT_freq = np.fft.fftfreq(len(pixel), d=period_s)
    return fourier_transform, FFT_freq


def load_collected_images(run_dir):
    """
    Loads a sequence of numpy array images from a directory.
    
    Args:
        run_dir (str): Directory path containing numbered image files
                      in format 'image_XXXX.npy'
    
    Returns:
        list: List of numpy arrays containing the loaded images
        
    Notes:
        - Expects images to be numbered sequentially starting from 0
        - File format must be 'image_0000.npy', 'image_0001.npy', etc.
        - Stops loading when it encounters first missing file number
        - Uses zero-padded 4-digit numbers (0000-9999)
    """
    images = []    # Initialize empty list to store images
    
    i = 0    # Counter for image numbering
    while True:
        # Construct path for current image number
        image_path = os.path.join(run_dir, f"image_{i:04d}.npy")
        
        # Exit loop if file doesn't exist (reached end of sequence)
        if not os.path.exists(image_path):
            break
            
        # Load numpy array and add to list
        images.append(np.load(image_path))
        i += 1
        
    return images


def run_FFT_scan(displacements_um, cam, pidevice, filename, velocity=None):
    if velocity == None:
        pixel_sum = collect_data_slow(displacements_um, cam, pidevice)
    else:
        pixel_sum = collect_data_fast(displacements_um, cam, pidevice, velocity)
    #plot_intensity(displacements_um, pixel_sum)
    fourier_transform, FFT_result_Hz = compute_FFT(displacements_um, pixel_sum)
    #plot_FFT(fourier_transform, FFT_result_Hz)
    save_data(displacements_um, pixel_sum, fourier_transform, FFT_result_Hz, filename)
    return displacements_um, pixel_sum, fourier_transform, FFT_result_Hz

def save_data(displacements, intensity, fft_result, fft_freq, filename):
    # Save the data
    np.savez(filename, 
            displacements=displacements, 
            intensity=intensity, 
            fft=fft_result, 
            fft_freq=fft_freq)
    print("Data saved to ", filename)

def compute_FFT_pixel_sum(displacements_um, pixel_sum):
    fourier_transform = np.fft.fft(pixel_sum) # Perform FFT of data
    period_s = (2 * (displacements_um[1] - displacements_um[0])* 1e-6) / c # Assuming uniform spacing
    FFT_freq = np.fft.fftfreq(len(pixel_sum), d=period_s)
    return fourier_transform, FFT_freq

def collect_data_fast(displacements_um, cam, pidevice, velocity):
    pidevice.VEL('1', velocity)
    data = displacements_um*0 # Array of zeros same size as displacements array
    for i_d, pos in enumerate(displacements_um):
            pidevice.MOV('1', pos)
            res = cam.GrabOne(100)  # Reduced timeout
            img = np.array(res.Array)
            res.Release() # Free up resources
            data[i_d] = np.sum(img[900:1100, 400:600])
    return data

def collect_data_slow(displacements_um, cam, pidevice):
    data = displacements_um*0 # Array of zeros same size as displacements array
    for i_d in range(len(displacements_um)):
        pitools.moveandwait(pidevice, '1',displacements_um[i_d]) # Move to each position in displacement array
        res = cam.GrabOne(1000) # Capture picture
        img = np.array(res.Array) # Store the picture's pixel values in an array
        #data[i_d] = img[1000,500]
        data[i_d] = np.sum(img[900:1100,400:600]) # Sum pixels in area of image with lots of fringes
    return data

                                                                                    
def plot_FFT_frequency(fourier_transform, FFT_freq):
    plt.figure(figsize=(10, 6))
    
    # Plot the magnitude spectrum
    plt.plot(FFT_freq, np.abs(fourier_transform))
    
    plt.title('Magnitude Spectrum of FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    # Set the x-axis to display only positive frequencies
    plt.xlim(0, max(FFT_freq))
    
    # Use a logarithmic scale for y-axis to better visualize the peaks
    plt.yscale('log')
    
    plt.grid(True)
    plt.show()


def plot_FFT_wavelength(fourier_transform, freq_Hz):
    # 3. Compute magnitude spectrum
    magnitude = np.abs(fourier_transform)

    # 4. Select only positive frequencies and their magnitudes
    positive_frequencies_Hz = freq_Hz[freq_Hz >= 0]
    positive_magnitude = magnitude[freq_Hz >= 0]

    wavelengths_m = c/positive_frequencies_Hz

    # 5. Plot the results
    plt.figure(figsize=(12, 6))

    # Plot magnitude spectrum of positive frequencies
    plt.plot(wavelengths_m*1e9, positive_magnitude)
    plt.title('Pixel Sum for [900:1100,400:600] Vs. Wavelength')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Pixel Sum')
    plt.xlim((100, 2000))
    #plt.xlim((630, 650))

    plt.tight_layout()
    plt.show()
    return positive_magnitude

def plot_data(x_axis, y_axis, x_axis_title=None, y_axis_title=None, title=None):
    plt.plot(x_axis, y_axis)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.title(title)
    plt.show()

def initialize_GSCDevice():
    # Create an instance of the pi controller
    pidevice = GCSDevice()

    # List available devices
    devices = pidevice.EnumerateUSB()
    print("Available devices:", devices)

    # Check if devices are available
    if devices:

        try:
            # Try to connect to the selected device
            pidevice.ConnectUSB(119020227)
            print("Connected successfully!")
            return pidevice

        except Exception as e:
            print("Failed to connect:", e)
            return None

    else:
        print("No devices found.")
        return None
    
def array_differences(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same length")
    
    differences = []
    for i in range(len(arr1)):
        difference = arr2[i] - arr1[i]
        differences.append(difference)
    return np.abs(differences)

# Example usage
array1 = [10, 15, 25, 35, 50]
array2 = [12, 19, 27, 33, 55]
result = array_differences(array1, array2)
print(result)  # Output: [2, 4, 2, -2, 5]
    
def initialize_pitools(pidevice):
    pitools.getaxeslist(pidevice, None) # Lists axes which are currently connected and available for control
    pitools.enableaxes(pidevice, '1') # Enables axis 1
    pitools.setservo(pidevice, '1', True) # Enables servo control for axis 1
    pitools.moveandwait(pidevice, '1', 500 ) # Moves to position 500 along axis 1

def initialize_cam():
    # get instance of the pylon TransportLayerFactory
    tlf = pylon.TlFactory.GetInstance()  # Initialize camera communications
    devices = tlf.EnumerateDevices()  # Show available devices
    cam = pylon.InstantCamera(tlf.CreateDevice(devices[0]))  # Creates an instance of the Basler camera
    cam.Open()
    res = cam.GrabOne(1000)
    return cam, res

def get_exposure_time(cam):
    try:
        exposure_time = cam.ExposureTime.GetValue()
        print(f"Current exposure time: {exposure_time} μs")
        return exposure_time
    except Exception as e:
        print(f"Error getting exposure time: {e}")
        return None
    
def set_exposure_time(cam, exposure_time_us):
    try:
        cam.ExposureTime.SetValue(exposure_time_us)
        print(f"Exposure time set to: {exposure_time_us} μs")
        
        # Verify the set exposure time
        actual_exposure = cam.ExposureTime.GetValue()
        print(f"Actual exposure time: {actual_exposure} μs")
        
        return actual_exposure
    except Exception as e:
        print(f"Error setting exposure time: {e}")
        return None


def close_cam(cam):
    cam.Close()

def show_picture(res):
    img = np.array(res.Array) # Store the image we grabbed as an array of pixel values
    # Show image
    plt.imshow(img)
    plt.show()
    res.Release()

# Define a function to parse the file
def parse_file(filename):
    time_data = []  # This will hold all the parsed rows
    position_data = []
    with open(filename, 'r') as file:
        for line in file:
            # Strip leading/trailing whitespace and split by tabs or spaces
            line = line.strip()
            if line:  # Skip empty lines
                # Split by one or more spaces/tabs
                parts = line.split()
                # Check if the row contains two items (time and position)
                if len(parts) == 2:
                    time = float(parts[0])
                    position = float(parts[1])
                    time_data.append(time)
                    position_data.append(position)
    return time_data, position_data