"""
ECG Signal Processing and Feature Extraction
Preprocess ECG data for multi-modal integration
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECGProcessor:
    """Process and extract features from ECG signals"""
    
    def __init__(self, sampling_rate: int = 360):
        self.sampling_rate = sampling_rate
        self.features = {}
    
    def preprocess_ecg(self, ecg_signal: np.ndarray, 
                      remove_baseline: bool = True,
                      filter_noise: bool = True) -> np.ndarray:
        """
        Preprocess ECG signal
        
        Args:
            ecg_signal: Raw ECG signal
            remove_baseline: Whether to remove baseline wander
            filter_noise: Whether to filter high-frequency noise
            
        Returns:
            Preprocessed ECG signal
        """
        processed_signal = ecg_signal.copy().astype(float)
        
        # Remove baseline wander using high-pass filter
        if remove_baseline:
            processed_signal = self._remove_baseline_wander(processed_signal)
        
        # Filter high-frequency noise
        if filter_noise:
            processed_signal = self._filter_noise(processed_signal)
        
        # Normalize signal
        processed_signal = self._normalize_signal(processed_signal)
        
        return processed_signal
    
    def _remove_baseline_wander(self, signal_data: np.ndarray) -> np.ndarray:
        """Remove baseline wander using high-pass filter"""
        # High-pass filter to remove frequencies below 0.5 Hz
        nyquist = 0.5 * self.sampling_rate
        cutoff = 0.5 / nyquist
        
        b, a = signal.butter(3, cutoff, btype='high')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def _filter_noise(self, signal_data: np.ndarray) -> np.ndarray:
        """Filter high-frequency noise"""
        # Low-pass filter to remove frequencies above 40 Hz
        nyquist = 0.5 * self.sampling_rate
        cutoff = 40 / nyquist
        
        b, a = signal.butter(3, cutoff, btype='low')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def _normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance"""
        normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
        return normalized
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Detect R-peaks in ECG signal"""
        # Use Pan-Tompkins algorithm for R-peak detection
        differentiated = np.diff(ecg_signal)
        squared = differentiated ** 2
        
        # Moving window integration
        window_size = int(0.15 * self.sampling_rate)  # 150ms window
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # Find peaks (simplified version)
        peaks, _ = signal.find_peaks(integrated, 
                                   height=np.mean(integrated) + 2*np.std(integrated),
                                   distance=int(0.3 * self.sampling_rate))  # 300ms min distance
        
        return peaks
    
    def extract_time_domain_features(self, ecg_signal: np.ndarray) -> Dict:
        """Extract time-domain features from ECG"""
        r_peaks = self.detect_r_peaks(ecg_signal)
        
        if len(r_peaks) < 2:
            logger.warning("Not enough R-peaks detected for feature extraction")
            return {}
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000  # Convert to ms
        
        features = {
            'mean_rr': np.mean(rr_intervals),
            'std_rr': np.std(rr_intervals),
            'mean_heart_rate': 60000 / np.mean(rr_intervals),  # bpm
            'rmssd': np.sqrt(np.mean(np.square(np.diff(rr_intervals)))),  # RMSSD
            'nn50': np.sum(np.abs(np.diff(rr_intervals)) > 50),  # NN50
            'pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100,
            'signal_energy': np.sum(ecg_signal ** 2),
            'signal_variance': np.var(ecg_signal),
            'signal_skewness': float(pd.Series(ecg_signal).skew()),
            'signal_kurtosis': float(pd.Series(ecg_signal).kurtosis()),
        }
        
        return features
    
    def extract_frequency_domain_features(self, ecg_signal: np.ndarray) -> Dict:
        """Extract frequency-domain features from ECG"""
        # Compute FFT
        n = len(ecg_signal)
        fft_vals = fft(ecg_signal)
        fft_freq = fftfreq(n, 1/self.sampling_rate)
        
        # Take only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_freq = fft_freq[positive_freq_idx]
        fft_vals = np.abs(fft_vals[positive_freq_idx])
        
        # Frequency bands for HRV analysis
        vlf_band = (0.003, 0.04)    # Very Low Frequency
        lf_band = (0.04, 0.15)      # Low Frequency  
        hf_band = (0.15, 0.4)       # High Frequency
        
        def band_power(freq_band):
            mask = (fft_freq >= freq_band[0]) & (fft_freq <= freq_band[1])
            return np.trapz(fft_vals[mask], fft_freq[mask])
        
        features = {
            'total_power': band_power((0.003, 0.4)),
            'vlf_power': band_power(vlf_band),
            'lf_power': band_power(lf_band),
            'hf_power': band_power(hf_band),
            'lf_hf_ratio': band_power(lf_band) / (band_power(hf_band) + 1e-8),
            'peak_frequency': fft_freq[np.argmax(fft_vals)],
            'spectral_entropy': self._spectral_entropy(fft_vals),
        }
        
        return features
    
    def _spectral_entropy(self, power_spectrum: np.ndarray) -> float:
        """Calculate spectral entropy"""
        # Normalize power spectrum to probability distribution
        power_normalized = power_spectrum / np.sum(power_spectrum)
        
        # Remove zeros to avoid log(0)
        power_normalized = power_normalized[power_normalized > 0]
        
        # Calculate spectral entropy
        entropy = -np.sum(power_normalized * np.log2(power_normalized))
        
        return entropy
    
    def extract_all_features(self, ecg_signal: np.ndarray) -> Dict:
        """Extract comprehensive set of ECG features"""
        time_features = self.extract_time_domain_features(ecg_signal)
        freq_features = self.extract_frequency_domain_features(ecg_signal)
        
        all_features = {**time_features, **freq_features}
        self.features = all_features
        
        return all_features

class ECGDataLoader:
    """Load and manage ECG datasets"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.ecg_signals = []
        self.labels = []
    
    def load_from_csv(self, file_path: str, signal_column: str = 'ecg_signal'):
        """Load ECG data from CSV file"""
        try:
            data = pd.read_csv(file_path)
            self.ecg_signals = data[signal_column].apply(
                lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else x
            ).tolist()
            self.labels = data['label'].values if 'label' in data.columns else None
            logger.info(f"Loaded {len(self.ecg_signals)} ECG signals")
        except Exception as e:
            logger.error(f"Error loading ECG data: {str(e)}")
            raise
    
    def preprocess_all_signals(self, processor: ECGProcessor) -> List[np.ndarray]:
        """Preprocess all loaded ECG signals"""
        processed_signals = []
        
        for i, signal in enumerate(self.ecg_signals):
            try:
                processed = processor.preprocess_ecg(signal)
                processed_signals.append(processed)
            except Exception as e:
                logger.warning(f"Error processing signal {i}: {str(e)}")
                processed_signals.append(signal)  # Keep original if processing fails
        
        return processed_signals
    
    def extract_features_batch(self, processor: ECGProcessor) -> pd.DataFrame:
        """Extract features from all ECG signals"""
        features_list = []
        
        for i, signal in enumerate(self.ecg_signals):
            try:
                features = processor.extract_all_features(signal)
                features['signal_id'] = i
                if self.labels is not None and i < len(self.labels):
                    features['label'] = self.labels[i]
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Error extracting features from signal {i}: {str(e)}")
        
        return pd.DataFrame(features_list)