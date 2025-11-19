import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Union
import matplotlib.pyplot as plt
import struct
from scipy import signal
import pandas as pd

def create_output_dir(output_path: str) -> None:
    """Create output directory if it doesn't exist."""
    Path(output_path).mkdir(parents=True, exist_ok=True)

def load_image(image_path: str) -> np.ndarray:
    """Load image from path."""
    return cv2.imread(image_path)

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to enhance contrast."""
    return cv2.equalizeHist(image)

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8,8)) -> np.ndarray:
    """Enhance contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def gaussian_smoothing(image: np.ndarray, kernel_size: Tuple[int, int] = (5,5)) -> np.ndarray:
    """Apply Gaussian smoothing to reduce noise."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def process_single_image(image_path: str, output_path: str, target_size: Tuple[int, int] = (224, 224)) -> None:
    """Process a single image with all preprocessing steps."""
    # Load image
    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert to grayscale
    image = convert_to_grayscale(image)

    # Apply preprocessing steps
    image = histogram_equalization(image)
    image = enhance_contrast(image)
    image = gaussian_smoothing(image)
    image = resize_image(image, target_size)

    # Save processed image
    output_file = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_file, image)

def process_all_images(input_dir: str, output_dir: str, target_size: Tuple[int, int] = (224, 224)) -> None:
    """Process all images in the input directory."""
    create_output_dir(output_dir)
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)
            process_single_image(input_path, output_dir, target_size)

def visualize_preprocessing(image_path: str) -> None:
    """Visualize preprocessing steps on a single image."""
    # Load original image
    original = load_image(image_path)
    if original is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert to grayscale
    original = convert_to_grayscale(original)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')

    # Histogram equalization
    hist_eq = histogram_equalization(original)
    axes[0,1].imshow(hist_eq, cmap='gray')
    axes[0,1].set_title('Histogram Equalization')
    axes[0,1].axis('off')

    # Contrast enhancement
    contrast = enhance_contrast(original)
    axes[1,0].imshow(contrast, cmap='gray')
    axes[1,0].set_title('Contrast Enhancement')
    axes[1,0].axis('off')

    # Gaussian smoothing
    smoothed = gaussian_smoothing(original)
    axes[1,1].imshow(smoothed, cmap='gray')
    axes[1,1].set_title('Gaussian Smoothing')
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.show()

# ============================================================================
# ECG Preprocessing Functions
# ============================================================================

def preprocess_jiachen_files(data_dir: str, labels_file: str) -> None:
    """Preprocess Jiachen's ECG files to fix naming inconsistencies"""
    data_path = Path(data_dir)
    labels_path = Path(labels_file)
    
    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    if not labels_path.exists():
        print(f"Labels file not found: {labels_file}")
        return
    
    print("Preprocessing Jiachen's files...")
    
    # Find all .dat files with _H suffix
    dat_files_with_h = list(data_path.glob('*_H.dat'))
    print(f"Found {len(dat_files_with_h)} files with _H suffix")
    
    renamed_count = 0
    for dat_file in dat_files_with_h:
        # Create new filename without _H
        new_dat_name = dat_file.name.replace('_H.dat', '.dat')
        new_dat_path = dat_file.parent / new_dat_name
        
        # Rename .dat file if target doesn't exist
        if not new_dat_path.exists():
            dat_file.rename(new_dat_path)
            renamed_count += 1
            print(f"Renamed: {dat_file.name} -> {new_dat_name}")
    
    print(f"Renamed {renamed_count} .dat files")
    
    # Update labels.csv to remove _H suffix from IDs
    try:
        df = pd.read_csv(labels_file)
        if 'id' in df.columns:
            # Remove _H suffix from IDs
            df['id'] = df['id'].astype(str).str.replace('_H', '', regex=False)
            
            # Save updated labels
            df.to_csv(labels_file, index=False)
            print(f"Updated {labels_file} to remove _H suffix from IDs")
            print(f"Updated labels shape: {df.shape}")
            print(f"Sample IDs: {df['id'].head().tolist()}")
        else:
            print("Warning: 'id' column not found in labels file")
    except Exception as e:
        print(f"Error updating labels file: {e}")

def read_physionet_data(dat_file, hea_file):
    """Read PhysioNet format ECG data (.dat + .hea files)"""
    print(f"Reading ECG data from {Path(dat_file).name}")
    return read_physionet_manual(dat_file, hea_file)

def read_physionet_manual(dat_file, hea_file):
    """Manual parsing of PhysioNet format"""
    # Read header file
    with open(hea_file, 'r') as f:
        lines = f.readlines()
    
    header_line = lines[0].strip().split()
    record_name = header_line[0]
    n_leads = int(header_line[1])
    sampling_rate = int(header_line[2])
    n_samples = int(header_line[3])
    
    print(f"Reading {record_name}: {n_leads} leads, {sampling_rate}Hz, {n_samples} samples")
    
    # Read binary data
    with open(dat_file, 'rb') as f:
        data = f.read()
    
    # Parse binary data (16-bit signed integers, interleaved by leads)
    n_values = len(data) // 2
    values = struct.unpack(f'<{n_values}h', data)
    
    try:
        expected_values = n_samples * n_leads
        if len(values) >= expected_values:
            values = values[:expected_values]
            signals = np.array(values).reshape(n_samples, n_leads)
        else:
            print(f"Warning: Expected {expected_values} values, got {len(values)}")
            padded_values = list(values) + [0] * (expected_values - len(values))
            signals = np.array(padded_values).reshape(n_samples, n_leads)
        
        return signals, sampling_rate, n_samples
    except Exception as e:
        print(f"Error reshaping data: {e}")
        return None, None, None

def process_ecg_signal(signals, sampling_rate, max_length=5000, start_threshold=500, end_threshold=150):
    """
    ECG processing with valid signal range detection and uniform sampling
    
    Args:
        signals: Raw ECG signals (samples, leads)
        sampling_rate: Original sampling rate
        max_length: Target length after uniform sampling
        start_threshold: Threshold to detect signal start (left side)
        end_threshold: Threshold to detect signal end (right side)
    
    Returns:
        processed_signal: 1D array of processed ECG signal
        valid_duration: Valid signal duration in seconds
        valid_range: Tuple of (start_idx, end_idx) for valid signal range
    """
    if signals is None:
        return None, None, None
    
    # Select first lead only
    if len(signals.shape) > 1:
        ecg_signal = signals[:, 0].astype(np.float32)
    else:
        ecg_signal = signals.astype(np.float32)
    
    # Find signal start with higher threshold
    start_indices = np.where(ecg_signal > start_threshold)[0]
    if len(start_indices) == 0:
        print(f"Warning: No signal above start threshold {start_threshold} found")
        return None, None, None
    
    # Find signal end with lower threshold
    end_indices = np.where(ecg_signal > end_threshold)[0]
    if len(end_indices) == 0:
        print(f"Warning: No signal above end threshold {end_threshold} found")
        return None, None, None
    
    # Get valid signal range
    start_idx = start_indices[0]  # First point > start_threshold
    end_idx = end_indices[-1]     # Last point > end_threshold
    valid_range = (start_idx, end_idx)
    
    # Extract valid signal segment
    valid_signal = ecg_signal[start_idx:end_idx+1]
    valid_length = len(valid_signal)
    valid_duration = valid_length / sampling_rate
    
    print(f"Valid signal range: [{start_idx}:{end_idx}] ({valid_length:,} samples, {valid_duration:.1f}s)")
    print(f"Original signal: {len(ecg_signal):,} samples, Valid signal: {valid_length:,} samples ({valid_length/len(ecg_signal)*100:.1f}%)")
    
    # Uniform sampling across valid signal range
    if valid_length > max_length:
        indices = np.linspace(0, valid_length - 1, max_length, dtype=int)
        processed_signal = valid_signal[indices]
    else:
        processed_signal = valid_signal
    
    # Remove DC offset only
    processed_signal = processed_signal - np.mean(processed_signal)
    
    return processed_signal, valid_duration, valid_range

def process_all_ecg_signals(input_dir: str, output_dir: str, max_length: int = 5000) -> int:
    """
    Process all ECG signals in the directory with uniform sampling
    
    Args:
        input_dir: Directory containing .dat and .hea files
        output_dir: Directory to save processed .npy files
        max_length: Target length for uniform sampling
    
    Returns:
        Number of successfully processed files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dat_files = list(input_path.glob('*.dat'))
    print(f"Processing {len(dat_files)} ECG files...")
    
    processed_count = 0
    
    for dat_file in dat_files:
        hea_file = Path(str(dat_file).replace('.dat', '.hea'))
        output_file = output_path / f"{dat_file.stem}.npy"
        
        # Skip if already processed
        if output_file.exists():
            continue
            
        if not hea_file.exists():
            print(f"✗ {dat_file.name}: header file missing")
            continue
        
        try:
            # Read ECG data
            raw_signals, sampling_rate, n_samples = read_physionet_data(dat_file, hea_file)
            
            if raw_signals is None:
                print(f"✗ {dat_file.name}: failed to read data")
                continue
            
            # Process signal
            processed_signal, valid_duration, valid_range = process_ecg_signal(
                raw_signals, sampling_rate, max_length
            )
            
            if processed_signal is None:
                print(f"✗ {dat_file.name}: failed to process")
                continue
            
            # Save processed signal
            np.save(output_file, processed_signal)
            processed_count += 1
            
            if processed_count <= 3:
                print(f"✓ {dat_file.name}: {len(processed_signal)} points, {valid_duration:.1f}s valid span")
                
        except Exception as e:
            print(f"✗ {dat_file.name}: {e}")
    
    if processed_count > 3:
        print(f"... and {processed_count - 3} more files processed")
    print(f"Total: {processed_count} ECG files processed")
    
    return processed_count

def visualize_ecg_preprocessing(dat_file, hea_file, max_length=5000, start_threshold=500, end_threshold=150):
    """Visualize ECG preprocessing: original vs valid range vs processed signal"""
    print(f"Visualizing: {Path(dat_file).name}")
    
    # Read raw data
    raw_signals, sampling_rate, n_samples = read_physionet_data(dat_file, hea_file)
    
    if raw_signals is None:
        print("Failed to read ECG data")
        return
    
    # Get original signal
    if len(raw_signals.shape) > 1:
        raw_complete = raw_signals[:, 0]  # First lead, complete signal
    else:
        raw_complete = raw_signals
    
    # Process signal (this will find valid range and process)
    processed_signal, valid_duration, valid_range = process_ecg_signal(
        raw_signals, sampling_rate, max_length, start_threshold, end_threshold
    )
    
    if processed_signal is None:
        print("Failed to process signal")
        return
    
    # Extract valid signal segment for comparison
    start_idx, end_idx = valid_range
    valid_signal = raw_complete[start_idx:end_idx+1]
    
    # Create time axes
    raw_time = np.arange(len(raw_complete)) / sampling_rate
    valid_time = np.arange(start_idx, end_idx+1) / sampling_rate
    processed_time = np.linspace(valid_time[0], valid_time[-1], len(processed_signal))
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Complete original signal with valid range highlighted
    print(f"Plotting complete original signal: {len(raw_complete):,} points...")
    axes[0].plot(raw_time, raw_complete, linewidth=0.3, color='gray', alpha=0.6, label='Complete signal')
    axes[0].plot(valid_time, valid_signal, linewidth=1.0, color='blue', label='Valid range')
    axes[0].axhline(y=start_threshold, color='red', linestyle='--', alpha=0.8, label=f'Start threshold (>{start_threshold})')
    axes[0].axhline(y=end_threshold, color='orange', linestyle='--', alpha=0.8, label=f'End threshold (>{end_threshold})')
    axes[0].set_title(f'Complete Original Signal ({len(raw_complete):,} points, {sampling_rate}Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    
    # Valid signal range only
    axes[1].plot(valid_time, valid_signal, linewidth=0.8, color='blue')
    axes[1].set_title(f'Valid Signal Range ({len(valid_signal):,} points, {valid_duration:.1f}s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    
    # Processed signal (uniform sampling from valid range)
    axes[2].plot(processed_time, processed_signal, linewidth=1.0, color='green')
    axes[2].set_title(f'Processed Signal ({len(processed_signal)} points, uniform sampling from valid range)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Original: {len(raw_complete):,} samples at {sampling_rate}Hz")
    print(f"Valid range: {len(valid_signal):,} samples ({valid_duration:.1f}s, {len(valid_signal)/len(raw_complete)*100:.1f}%)")
    print(f"Processed: {len(processed_signal)} samples from valid range")
    print(f"Valid signal compression ratio: {len(valid_signal)/len(processed_signal):.1f}:1")

# ============================================================================
# Voice Signal Preprocessing Functions
# ============================================================================

def read_voice_data(txt_file: str, hea_file: str) -> Tuple[np.ndarray, int]:
    """Read voice signal data from .txt file and sampling rate from .hea file"""
    # Read sampling rate from header
    try:
        with open(hea_file, 'r', encoding='utf-8', errors='ignore') as f:
            header_line = f.readline().strip()
            parts = header_line.split()
            record_name = parts[0]
            n_leads = int(parts[1])
            sampling_rate = int(parts[2])
            n_samples = int(parts[3])
    except Exception as e:
        print(f"Error reading header file {hea_file}: {e}")
        return None, None
    
    print(f"Reading {record_name}: {n_leads} leads, {sampling_rate}Hz, {n_samples} samples")
    
    # Read signal data from txt file with error handling
    voice_signal = []
    try:
        # Try UTF-8 first
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        value = float(line)
                        if not np.isnan(value) and not np.isinf(value):
                            voice_signal.append(value)
                        else:
                            print(f"Warning: Invalid value at line {line_num}: {line}")
                            voice_signal.append(0.0)
                    except ValueError:
                        print(f"Warning: Could not convert line {line_num} to float: {line}")
                        voice_signal.append(0.0)
    except UnicodeDecodeError:
        try:
            # Try latin-1 if UTF-8 fails
            with open(txt_file, 'r', encoding='latin-1') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            value = float(line)
                            if not np.isnan(value) and not np.isinf(value):
                                voice_signal.append(value)
                            else:
                                print(f"Warning: Invalid value at line {line_num}: {line}")
                                voice_signal.append(0.0)
                        except ValueError:
                            print(f"Warning: Could not convert line {line_num} to float: {line}")
                            voice_signal.append(0.0)
        except UnicodeDecodeError:
            # Try with no encoding specified (binary mode)
            with open(txt_file, 'r', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            value = float(line)
                            if not np.isnan(value) and not np.isinf(value):
                                voice_signal.append(value)
                            else:
                                print(f"Warning: Invalid value at line {line_num}: {line}")
                                voice_signal.append(0.0)
                        except ValueError:
                            print(f"Warning: Could not convert line {line_num} to float: {line}")
                            voice_signal.append(0.0)
    except Exception as e:
        print(f"Error reading txt file {txt_file}: {e}")
        return None, None
    
    if len(voice_signal) == 0:
        print(f"Error: No valid data read from {txt_file}")
        return None, None
    
    voice_signal = np.array(voice_signal)
    
    # Check for NaN or Inf values
    if np.isnan(voice_signal).any() or np.isinf(voice_signal).any():
        print(f"Warning: NaN/Inf values found in {txt_file}, replacing with zeros")
        voice_signal = np.nan_to_num(voice_signal, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure we have the expected number of samples
    if len(voice_signal) >= n_samples:
        voice_signal = voice_signal[:n_samples]
    else:
        print(f"Warning: Expected {n_samples} values, got {len(voice_signal)}")
    
    return voice_signal, sampling_rate

def read_voice_metadata(info_file: str) -> Dict[str, Union[str, int, float]]:
    """Read voice signal metadata from info file"""
    metadata = {}
    
    with open(info_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if ':' in line and not line.startswith('#'):
            # Split on first colon only to handle values that might contain colons
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                # Skip empty values
                if not value or value == '':
                    continue
                
                # Convert numeric values
                if value.replace('.', '').replace(',', '').replace('-', '').isdigit():
                    # Handle comma as decimal separator (e.g., "1,5" -> 1.5)
                    if ',' in value and '.' not in value:
                        value = value.replace(',', '.')
                    value = float(value)
                    # Convert to int if it's a whole number
                    if value.is_integer():
                        value = int(value)
                elif value.lower() in ['yes', 'no', 'true', 'false']:
                    # Keep boolean-like values as strings
                    value = value.lower()
                elif value.upper() == 'NU':
                    # Handle missing values
                    value = None
                else:
                    # Keep other values as strings
                    value = value
                
                metadata[key] = value
    
    return metadata

def preprocess_voice_signal(
    voice_signal: np.ndarray, 
    sampling_rate: int, 
    target_length: int = 5000,
    low_freq: float = 80.0,
    high_freq: float = 3400.0,
    gaussian_sigma: float = 1.0  # 添加高斯平滑参数
) -> np.ndarray:
    """
    Preprocess voice signal with filtering, normalization, and uniform sampling.
    
    Args:
        voice_signal: Input voice signal
        sampling_rate: Sampling rate in Hz
        target_length: Target length for uniform sampling
        low_freq: Low frequency cutoff for bandpass filter
        high_freq: High frequency cutoff for bandpass filter
        gaussian_sigma: Standard deviation for Gaussian smoothing
        
    Returns:
        Preprocessed voice signal
    """
    # Check for invalid input
    if voice_signal is None or len(voice_signal) == 0:
        print("Error: Empty or None voice signal")
        return np.zeros(target_length)
    
    if np.isnan(voice_signal).any() or np.isinf(voice_signal).any():
        print("Warning: Input signal contains NaN or Inf values")
        voice_signal = np.nan_to_num(voice_signal, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Remove DC component
    voice_signal = voice_signal - np.mean(voice_signal)
    
    # Check after DC removal
    if np.isnan(voice_signal).any() or np.isinf(voice_signal).any():
        print("Warning: NaN/Inf values after DC removal")
        voice_signal = np.nan_to_num(voice_signal, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Apply bandpass filter
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Check filter parameters
    if low >= high or low <= 0 or high >= 1:
        print(f"Warning: Invalid filter parameters: low={low}, high={high}")
        low = 0.01
        high = 0.99
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        voice_signal = signal.filtfilt(b, a, voice_signal)
    except Exception as e:
        print(f"Error in bandpass filtering: {e}")
        # Skip filtering if it fails
    
    # Check after filtering
    if np.isnan(voice_signal).any() or np.isinf(voice_signal).any():
        print("Warning: NaN/Inf values after filtering")
        voice_signal = np.nan_to_num(voice_signal, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Apply Gaussian smoothing
    if gaussian_sigma > 0:
        try:
            # Create Gaussian kernel
            kernel_size = int(3 * gaussian_sigma) * 2 + 1  # Ensure odd size
            kernel_size = max(3, min(kernel_size, 21))  # Limit kernel size between 3 and 21
            x = np.arange(-kernel_size//2, kernel_size//2 + 1)
            kernel = np.exp(-(x**2) / (2 * gaussian_sigma**2))
            kernel = kernel / np.sum(kernel)  # Normalize
            
            # Apply smoothing
            voice_signal = np.convolve(voice_signal, kernel, mode='same')
        except Exception as e:
            print(f"Error in Gaussian smoothing: {e}")
            # Skip smoothing if it fails
    
    # Check after smoothing
    if np.isnan(voice_signal).any() or np.isinf(voice_signal).any():
        print("Warning: NaN/Inf values after smoothing")
        voice_signal = np.nan_to_num(voice_signal, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Normalize
    if np.std(voice_signal) > 0:
        voice_signal = (voice_signal - np.mean(voice_signal)) / np.std(voice_signal)
    else:
        print("Warning: Zero standard deviation, skipping normalization")
    
    # Check after normalization
    if np.isnan(voice_signal).any() or np.isinf(voice_signal).any():
        print("Warning: NaN/Inf values after normalization")
        voice_signal = np.nan_to_num(voice_signal, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Uniform sampling
    if len(voice_signal) != target_length:
        indices = np.linspace(0, len(voice_signal) - 1, target_length, dtype=int)
        voice_signal = voice_signal[indices]
    
    # Final check
    if np.isnan(voice_signal).any() or np.isinf(voice_signal).any():
        print("Warning: Final signal contains NaN/Inf values, replacing with zeros")
        voice_signal = np.nan_to_num(voice_signal, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return voice_signal

def process_voice_file(
    txt_file: str, 
    hea_file: str, 
    info_file: str,
    output_dir: str,
    target_length: int = 5000  # Changed to 5000 points
) -> Dict[str, Union[str, np.ndarray, Dict]]:
    """
    Process a single voice file and save preprocessed data.
    
    Args:
        txt_file: Path to .txt file (ASCII signal data)
        hea_file: Path to .hea file (header with sampling rate)
        info_file: Path to -info.txt file
        output_dir: Output directory for processed data
        target_length: Target length of processed signal (number of points)
    
    Returns:
        Dictionary containing processed data and metadata
    """
    # Read voice data
    signal, sampling_rate = read_voice_data(txt_file, hea_file)
    
    if signal is None or sampling_rate is None:
        print(f"Error: Failed to read voice data from {txt_file}")
        return None
    
    # Read metadata
    try:
        metadata = read_voice_metadata(info_file)
    except Exception as e:
        print(f"Warning: Error reading metadata from {info_file}: {e}")
        metadata = {}
    
    # Preprocess signal
    processed_signal = preprocess_voice_signal(
        signal, sampling_rate, target_length
    )
    
    if processed_signal is None:
        print(f"Error: Failed to preprocess signal from {txt_file}")
        return None
    
    # Save processed data
    record_name = Path(txt_file).stem
    output_file = Path(output_dir) / f"{record_name}.npy"
    np.save(output_file, processed_signal)
    
    return {
        'record_name': record_name,
        'signal': processed_signal,
        'metadata': metadata,
        'output_file': str(output_file)
    }

def process_all_voice_signals(
    input_dir: str, 
    output_dir: str, 
    target_length: int = 5000  # Changed to 5000 points
) -> int:
    """
    Process all voice files in the input directory.
    
    Args:
        input_dir: Input directory containing voice files
        output_dir: Output directory for processed data
        target_length: Target length of processed signals (number of points)
    
    Returns:
        Number of successfully processed files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read RECORDS file
    records_file = input_path / "RECORDS"
    if not records_file.exists():
        print(f"RECORDS file not found in {input_dir}")
        return 0
    
    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f.readlines()]
    
    processed_count = 0
    
    for record_name in record_names:
        txt_file = input_path / f"{record_name}.txt"
        hea_file = input_path / f"{record_name}.hea"
        info_file = input_path / f"{record_name}-info.txt"
        
        # Check if all required files exist
        if not all(f.exists() for f in [txt_file, hea_file, info_file]):
            print(f"Missing files for {record_name}")
            continue
        
        try:
            result = process_voice_file(
                str(txt_file), str(hea_file), str(info_file),
                str(output_path), target_length
            )
            if result is not None:
                processed_count += 1
                print(f"Processed {record_name}")
            else:
                print(f"Failed to process {record_name}")
        except Exception as e:
            print(f"Error processing {record_name}: {e}")
    
    print(f"Total: {processed_count} voice files processed")
    return processed_count

def create_voice_labels_file(
    input_dir: str, 
    output_file: str,
    target_variable: str = 'VHI Score'
) -> pd.DataFrame:
    """
    Create labels file for voice data based on metadata.
    
    Args:
        input_dir: Input directory containing voice files
        output_file: Output CSV file for labels
        target_variable: Target variable from metadata to use as label
                        Options: 'VHI Score', 'RSI Score', 'Age', 'Diagnosis'
    
    Returns:
        DataFrame with all metadata fields, target variable renamed to 'y'
    """
    input_path = Path(input_dir)
    
    # Read RECORDS file
    records_file = input_path / "RECORDS"
    if not records_file.exists():
        print(f"RECORDS file not found in {input_dir}")
        return pd.DataFrame()
    
    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f.readlines()]
    
    labels_data = []
    available_variables = set()
    
    for record_name in record_names:
        info_file = input_path / f"{record_name}-info.txt"
        
        if not info_file.exists():
            print(f"Info file not found for {record_name}")
            continue
        
        try:
            metadata = read_voice_metadata(str(info_file))
            available_variables.update(metadata.keys())
            
            # Check if target variable exists
            if target_variable not in metadata:
                print(f"Target variable '{target_variable}' not found for {record_name}")
                continue
            
            # Skip if target variable is None (missing data)
            if metadata[target_variable] is None:
                print(f"Missing {target_variable} for {record_name}")
                continue
            
            # Create row with all metadata
            row_data = {
                'id': record_name,
                **metadata  # Include all metadata fields
            }
            
            # Rename target variable to 'y'
            if target_variable in row_data:
                row_data['y'] = row_data.pop(target_variable)
            
            labels_data.append(row_data)
            
        except Exception as e:
            print(f"Error reading metadata for {record_name}: {e}")
    
    # Create DataFrame and save
    labels_df = pd.DataFrame(labels_data)
    
    if not labels_df.empty:
        # Reorder columns to put 'id' and 'y' first
        columns = ['id', 'y'] + [col for col in labels_df.columns if col not in ['id', 'y']]
        labels_df = labels_df[columns]
        
        labels_df.to_csv(output_file, index=False)
        
        print(f"Created labels file with {len(labels_df)} samples")
        print(f"Target variable: {target_variable} (renamed to 'y')")
        print(f"All available fields: {list(labels_df.columns)}")
        
        # Print label distribution
        if target_variable == 'Diagnosis':
            print(f"Diagnosis distribution:")
            print(labels_df['y'].value_counts())
        else:
            print(f"Label distribution:\n{labels_df['y'].describe()}")
        
        # Print sample of the data
        print(f"\nSample of labels data:")
        print(labels_df.head())
        
    else:
        print(f"No valid labels found for target variable: {target_variable}")
    
    return labels_df

def visualize_voice_preprocessing(
    txt_file: str, 
    hea_file: str,
    target_length: int = 5000,
    gaussian_sigma: float = 1.0
) -> None:
    """
    Visualize voice signal preprocessing steps.
    
    Args:
        txt_file: Path to .txt file (ASCII signal data)
        hea_file: Path to .hea file (header with sampling rate)
        target_length: Target length for processing (number of points)
        gaussian_sigma: Standard deviation for Gaussian smoothing
    """
    # Read voice data using the updated function
    voice_signal, sampling_rate = read_voice_data(txt_file, hea_file)
    
    # Calculate the middle 5% region
    total_points = len(voice_signal)
    middle_start = int(total_points * 0.475)  # Start at 47.5%
    middle_end = int(total_points * 0.525)    # End at 52.5%
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # Original signal (middle 5%)
    time_axis_original = np.arange(len(voice_signal)) / sampling_rate
    time_middle = time_axis_original[middle_start:middle_end]
    signal_middle = voice_signal[middle_start:middle_end]
    
    axes[0].plot(time_middle, signal_middle)
    axes[0].set_title('Original Voice Signal (Middle 5%)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    # After DC removal and bandpass filtering
    voice_signal_no_dc = voice_signal - np.mean(voice_signal)
    nyquist = sampling_rate / 2
    low = 80.0 / nyquist
    high = 3400.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    voice_signal_filtered = signal.filtfilt(b, a, voice_signal_no_dc)
    
    signal_filtered_middle = voice_signal_filtered[middle_start:middle_end]
    axes[1].plot(time_middle, signal_filtered_middle)
    axes[1].set_title('After DC Removal and Bandpass Filter (80-3400 Hz) - Middle 5%')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)
    
    # After Gaussian smoothing
    if gaussian_sigma > 0:
        kernel_size = int(3 * gaussian_sigma) * 2 + 1
        kernel_size = max(3, min(kernel_size, 21))
        x = np.arange(-kernel_size//2, kernel_size//2 + 1)
        kernel = np.exp(-(x**2) / (2 * gaussian_sigma**2))
        kernel = kernel / np.sum(kernel)
        voice_signal_smoothed = np.convolve(voice_signal_filtered, kernel, mode='same')
    else:
        voice_signal_smoothed = voice_signal_filtered
    
    signal_smoothed_middle = voice_signal_smoothed[middle_start:middle_end]
    axes[2].plot(time_middle, signal_smoothed_middle)
    axes[2].set_title(f'After Gaussian Smoothing (σ={gaussian_sigma}, kernel_size={kernel_size}) - Middle 5%')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True)
    
    # Final preprocessed signal (normalized and uniformly sampled)
    if np.std(voice_signal_smoothed) > 0:
        voice_signal_normalized = (voice_signal_smoothed - np.mean(voice_signal_smoothed)) / np.std(voice_signal_smoothed)
    else:
        voice_signal_normalized = voice_signal_smoothed
    
    # Uniform sampling
    indices = np.linspace(0, len(voice_signal_normalized) - 1, target_length, dtype=int)
    voice_signal_final = voice_signal_normalized[indices]
    
    # Calculate time axis for final signal and get middle 5%
    original_duration = len(voice_signal) / sampling_rate
    time_axis_final = np.linspace(0, original_duration, target_length)
    final_middle_start = int(target_length * 0.475)
    final_middle_end = int(target_length * 0.525)
    time_final_middle = time_axis_final[final_middle_start:final_middle_end]
    signal_final_middle = voice_signal_final[final_middle_start:final_middle_end]
    
    axes[3].plot(time_final_middle, signal_final_middle)
    axes[3].set_title(f'Final Preprocessed Signal (Normalized, {target_length} points) - Middle 5%')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Amplitude')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print processing statistics
    print(f"Processing Statistics:")
    print(f"Original signal length: {len(voice_signal)} points")
    print(f"Original duration: {len(voice_signal) / sampling_rate:.2f} seconds")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Gaussian kernel size: {kernel_size}")
    print(f"Final signal length: {len(voice_signal_final)} points")
    print(f"Signal range after processing: [{voice_signal_final.min():.3f}, {voice_signal_final.max():.3f}]")
    print(f"Displayed region: Middle 5% ({middle_start}-{middle_end} points)")

