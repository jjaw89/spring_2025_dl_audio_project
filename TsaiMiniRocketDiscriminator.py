import multiprocessing
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# MiniRocket Discriminator using tsai library
class TsaiMiniRocketDiscriminator(nn.Module):
    def __init__(
        self,
        freq_bins=256,
        time_frames=256,
        num_kernels=10000,  # number of convolutional kernels
        hidden_dim=1024,    # Increased to handle larger feature dimension
        output_dim=1,
        accompaniment = True   # whether or not we feed accompaniment
    ):
        num_cores = multiprocessing.cpu_count()
        print("Number of CPU cores:", num_cores)
        minirocket_n_jobs = num_cores - 4

        super(TsaiMiniRocketDiscriminator, self).__init__()

        # This is the mini rocket transformer which extracts features
        self.rocket = MiniRocketMultivariate(num_kernels=num_kernels, n_jobs=minirocket_n_jobs)
        # tsai's miniRocketClassifier is implemented with MiniRocketMultivariate as well
        self.fitted = False   # fit before training
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        self.num_kernels = num_kernels
        self.accompaniment = accompaniment

        # For 2D data handling - process each sample with proper dimensions
        self.example_input = np.zeros((1, freq_bins, time_frames))

        self.feature_dim = num_kernels  # For vocals + accompaniment

        classifier_input_dim = 9996

        # Example feature reducing layers
        self.classifier = nn.Sequential(
            # First reduce the massive dimension to something manageable
            # nn.Dropout(0.3),
            spectral_norm(nn.Linear(classifier_input_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Second hidden layer
            # nn.Dropout(0.3),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),

            # Final classification layer
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def fit_rocket(self, vocals, accompaniment = None):
        """
            Fit MiniRocket with just one piece of vocal training data (not the entire training dataset)
        """
        if not self.fitted:
            try:
                if accompaniment:
                    spectrograms = torch.cat((vocals, accompaniment), dim = 1)
                else:
                    spectrograms = vocals

                # Reshape for MiniRocket - it expects (n_instances, n_dimensions, series_length)
                # flatten the freq_bins dimension to create a multivariate time series
                batch_size = spectrograms.shape[0]

                # Convert first to numpy for sktime processing
                sample_data = spectrograms.detach().cpu().numpy()
                # print(sample_data.shape)
                # Reshape to sktime's expected format - reduce to single sample for fitting
                # sample_data = sample_data[:, 0]  # Take one sample, remove channel dim

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
            spec_np = spectrogram.detach().cpu().numpy()

            # Remove channel dimension expected by sktime
            # print(spec_np.shape)
            # spec_np = spec_np[:, 0]  # [batch_size, freq_bins, time_frames]
            # print(spec_np.shape)

            # This step extracts features using the convolutional kernels, numbers specified by num_kernels
            # print("1")
            features = self.rocket.transform(spec_np)
            # print("2")
            # Convert back to torch tensor
            # print("features:", features.shape)
            # print(features.head())
            features_tensor = torch.tensor(features.values).to(spectrogram.device)
            # print("features:", features.shape)
            # print("3")
            return features_tensor

        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Return zeros as fallback
            return torch.zeros((spectrogram.shape[0], self.num_kernels),
                              device=spectrogram.device)

    def forward(self, vocals, accompaniment = None):
        """
        Forward pass of the discriminator

        Args:
            vocals: Spectrograms of shape [batch_size, channels, freq_bins, time_frames]
            accompaniment: Spectrograms of shape [batch_size, channels, freq_bins, time_frames]
        """
        # Extract features from both spectrograms
        # start_time = time()
#        vocal_features = self.extract_features(vocals)

        if self.accompaniment:
          input = torch.cat((vocals, accompaniment), dim=1)
          # print(combined_features.size())
        else:
          input = vocals

        # Classify as real/fake
        # print(combined_features.size())
        output_features = self.extract_features(input)
        validity = self.classifier(output_features)

        return validity

