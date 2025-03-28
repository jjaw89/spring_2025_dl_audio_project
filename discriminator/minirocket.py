import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# placeholder for waveunet generator
class WaveUNetGenerator(nn.Module):
    # implementation of waveunet model which I assume serves as generator
    # Based on meeting notes, generator should output two spectrogram, each of the size 256(time) x 256(frequency bins)
    pass

# MiniRocket Discriminator using tsai library
class TsaiMiniRocketDiscriminator(nn.Module):
    def __init__(
        self,
        freq_bins=256,
        time_frames=256,
        num_kernels=10000,  # number of convolutional kernels
        hidden_dim=1024,    # Increased to handle larger feature dimension
        output_dim=1
    ):
        super(TsaiMiniRocketDiscriminator, self).__init__()
        
        # This is the mini rocket transformer which extracts features
        self.rocket = MiniRocketMultivariate(num_kernels=num_kernels)  
        # tsai's miniRocketClassifier is implemented with MiniRocketMultivariate as well
        self.fitted = False   # fit before training
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        self.num_kernels = num_kernels
        
        # For 2D data handling - process each sample with proper dimensions
        self.example_input = np.zeros((1, freq_bins, time_frames))
        
        feature_dim = num_kernels * 2  # For vocals + accompaniment
        
        # Example feature reducing layers
        self.classifier = nn.Sequential(
            # First reduce the massive dimension to something manageable
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Final classification layer
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def fit_rocket(self, spectrograms):
        """
            Fit MiniRocket with just one piece of vocal training data (not the entire training dataset)
        """
        if not self.fitted:
            try:
                # Reshape for MiniRocket - it expects (n_instances, n_dimensions, series_length)
                # flatten the freq_bins dimension to create a multivariate time series
                batch_size = spectrograms.shape[0]
                
                # Convert first to numpy for sktime processing
                sample_data = spectrograms.cpu().numpy()
                
                # Reshape to sktime's expected format - reduce to single sample for fitting
                sample_data = sample_data[:1, 0]  # Take one sample, remove channel dim
                
                # Fit on this sample 
                self.rocket.fit(sample_data)
                self.fitted = True
                
                # Test transform to get feature dimension
                test_transform = self.rocket.transform(sample_data)
                self.feature_dim = test_transform.shape[1]
                
                print(f"MiniRocket fitted. Feature dimension: {self.feature_dim}")
                
            except Exception as e:
                print(f"Error fitting MiniRocket: {e}")
                # Use a fallback if fitting fails
                self.fitted = True  # Mark as fitted to avoid repeated attempts
    
    def extract_features(self, spectrogram):
        """Extract MiniRocket features from a spectrogram"""
        try:
            # Ensure rocket is fitted
            if not self.fitted:
                self.fit_rocket(spectrogram)
            
            # Convert to numpy for sktime
            spec_np = spectrogram.cpu().numpy()
            
            # Remove channel dimension expected by sktime
            spec_np = spec_np[:, 0]  # [batch_size, freq_bins, time_frames]
            
            # This step extracts features using the convolutional kernels, numbers specified by num_kernels
            features = self.rocket.transform(spec_np)
            
            # Convert back to torch tensor
            features = torch.tensor(features, dtype=torch.float32).to(spectrogram.device)
            
            return features
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Return zeros as fallback
            return torch.zeros((spectrogram.shape[0], self.num_kernels), 
                              device=spectrogram.device)
    
    def forward(self, vocals, accompaniment):
        """
        Forward pass of the discriminator
        
        Args:
            vocals: Spectrograms of shape [batch_size, channels, freq_bins, time_frames]
            accompaniment: Spectrograms of shape [batch_size, channels, freq_bins, time_frames]
        """
        # Extract features from both spectrograms
        vocal_features = self.extract_features(vocals)
        accomp_features = self.extract_features(accompaniment)
        
        # Concatenate features (conditional GAN)
        combined_features = torch.cat([vocal_features, accomp_features], dim=1)
        
        # Classify as real/fake
        validity = self.classifier(combined_features)
        
        return validity

