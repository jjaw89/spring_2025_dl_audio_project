import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class SingingDataset(Dataset):

  def __init__(self, musDB, window_size = 256, step_size = 128):
    self.mel_specs = torch.zeros(1, 128, window_size)
    self.sample_rates = torch.tensor([0])

    num_songs = 0

    for track in musDB:
      stems, rate = track.stems, track.rate

      num_songs += 1

      # load the vocal
      vocal = librosa.to_mono(np.transpose(stems[4]))

      # compute log mel spectrogram and convert to pytorch tensor
      mel_spec = torch.from_numpy(self._mel_spectrogram(vocal, rate))

      start_ndx = 0
      for step in range(window_size // step_size):
        cropped_mel_spec = mel_spec[:, start_ndx:]
        num_slices = cropped_mel_spec.shape[1] // window_size

        # chop off the last bit so that number of stft steps is a multiple of window_size
        cropped_mel_spec = cropped_mel_spec[:, 0:num_slices*window_size]

        # reshape tensors into chunks of size 128x(window_size)
        # first dimension is number of chunks
        cropped_mel_spec = torch.transpose(torch.reshape(cropped_mel_spec, (128, num_slices, window_size)), 0, 1)

        # concatenate to the big tensor
        self.mel_specs = torch.cat((self.mel_specs, cropped_mel_spec), 0)
        self.sample_rates = torch.cat((self.sample_rates, torch.full((num_slices,), rate)), 0)


    if num_songs % 5 == 0:
        print(str(num_songs) + " songs processed; produced " + str(self.mel_specs.shape[0]) + " spectrograms")

    # remove the all zeros slice that we initialized with
    self.mel_specs = self.mel_specs[1: , : , :]
    self.sample_rates = self.sample_rates[1:]

  def __len__(self):
    return self.mel_specs.shape[0]

  def __getitem__(self, ndx):
    # returns tuple (mel spectrogram of accompaniment, mel spectrogram of vocal, rate)
    return self.mel_specs[ndx], self.sample_rates[ndx]

  def _mel_spectrogram(self, audio, rate):
    # compute the log-mel-spectrogram of the audio at the given sample rate
    return librosa.power_to_db(librosa.feature.melspectrogram(y = audio, sr = rate))