import torch
import librosa
import numpy as np
import os
from torch.utils.data import Dataset

class LibriSpeechDataset(Dataset):

    def __init__(self, path, window_size = 256, step_size = 128, num_specs = 7647*2):
        self.mel_specs = self.mel_specs = torch.zeros(1, 128, window_size)
        self.sample_rates = torch.tensor([0])
        
        num_files_opened = 0

        for speaker_dir in os.listdir(path):
            speaker_path = path + "/" + speaker_dir
            for chapter_dir in os.listdir(speaker_path):
                chapter_path = speaker_path + "/" + chapter_dir
                for file in os.listdir(chapter_path):
                    # checks file extension and stops when we hit desired number of spectrograms (num_specs)
                    if file.endswith('.flac') and self.mel_specs.shape[0] - 1 < num_specs:
                        # get audio file and convert to log mel spectrogram
                        speech, rate = librosa.load(chapter_path + "/" + file, sr = 44100)
                        mel_spec = torch.from_numpy(self._mel_spectrogram(speech, rate))
                        start_ndx = 0
                        
                        num_files_opened += 1
                        
                        for step in range(window_size // step_size):
                            cropped_mel_spec = mel_spec[:, start_ndx:]
                            
                            # Saves the total number of 128 x (window_size) spectrograms
                            num_slices = cropped_mel_spec.shape[1] // window_size

                            # chop off the last bit so that number of stft steps is a multiple of window_size
                            cropped_mel_spec = cropped_mel_spec[ : , 0 : num_slices*window_size]
    
                            # reshape the tensor to have many spectrograms of size 128 x (steps)
                            cropped_mel_spec = torch.transpose(torch.reshape(cropped_mel_spec, (128, num_slices, window_size)), 0, 1)

                            # concatenate tensor to the full tensor in the Dataset object
                            self.mel_specs = torch.cat((self.mel_specs, cropped_mel_spec), 0)
                            self.sample_rates = torch.cat((self.sample_rates, torch.full((num_slices,), rate)), 0)
                            
                            # increment start_ndx
                            start_ndx += step_size
                            
                        
                        if num_files_opened % 50 == 0:
                            print("opened " + str(num_files_opened) + " files and produced " + str(self.mel_specs.shape[0]) + " spectrograms")
                            

        # chop off the zero layer we initialized with
        self.mel_specs = self.mel_specs[1:]
        self.sample_rates = self.sample_rates[1:]

    def __len__(self):
        return self.mel_specs.shape[0]

    def __getitem__(self, ndx):
        return self.mel_specs[ndx], self.sample_rates[ndx]

    def _mel_spectrogram(self, audio, rate):
        # compute the log-mel-spectrogram of the audio at the given sample rate
        return librosa.power_to_db(librosa.feature.melspectrogram(y = audio, sr = rate))