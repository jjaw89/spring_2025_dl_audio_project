import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class MusdbDataset(Dataset):

  def __init__(self, musDB, steps = 256):
    self.mel_specs = torch.zeros(1, 2, 128, steps)
    self.sample_rates = torch.tensor([0])

    for track in musDB:
      stems, rate = track.stems, track.rate

      # separate the vocal from other instruments and conver to mono signal
      audio_novocal = librosa.to_mono(np.transpose(stems[1] + stems[2] + stems[3]))
      audio_vocal = librosa.to_mono(np.transpose(stems[4]))

      # compute log mel spectrogram and convert to pytorch tensor
      logmelspec_novocal = torch.from_numpy(self._mel_spectrogram(audio_novocal, rate))
      logmelspec_vocal = torch.from_numpy(self._mel_spectrogram(audio_vocal, rate))

      num_slices = logmelspec_novocal.shape[1] // steps

      # chop off the last bit so that number of stft steps is a multiple of step size
      logmelspec_novocal = logmelspec_novocal[0:128 , 0:num_slices*steps]
      logmelspec_vocal = logmelspec_vocal[0:128, 0:num_slices*steps]

      logmelspec_novocal = torch.reshape(logmelspec_novocal, (num_slices, 128, steps))
      logmelspec_vocal = torch.reshape(logmelspec_vocal, (num_slices, 128, steps))

      # unsqueeze and concatenate these tensors. Then concatenate to the big tensor
      logmels = torch.cat((logmelspec_novocal.unsqueeze(1), logmelspec_vocal.unsqueeze(1)), 1)
      self.mel_specs = torch.cat((self.mel_specs, logmels), 0)
      self.sample_rates = torch.cat((self.sample_rates, torch.Tensor([rate])), 0)
    
    # remove the all zeros slice that we initialized with
    self.mel_specs = self.mel_specs[1: , : , : , :]
    self.sample_rates = self.sample_rates[1:]

  def __len__(self):
    return self.mel_specs.shape[0]

  def __getitem__(self, ndx):
    # returns tuple (mel spectrogram of accompaniment, mel spectrogram of vocal, rate)
    return self.mel_specs[ndx, 0], self.mel_specs[ndx, 1], self.sample_rates[ndx]

  def _mel_spectrogram(self, audio, rate):
    # compute the log-mel-spectrogram of the audio at the given sample rate
    return librosa.power_to_db(librosa.feature.melspectrogram(y = audio, sr = rate))
