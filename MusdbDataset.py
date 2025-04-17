import stempeg
import musdb
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class MusdbDataset(Dataset):

  def __init__(self, musDB, window_size = 256, step_size = 128):
    self.mel_specs = torch.zeros(1, 2, 128, window_size)
    self.sample_rates = torch.tensor([0])

    num_songs = 0

    for track in musDB:
      stems, rate = track.stems, track.rate

      num_songs += 1

      # separate the vocal from other instruments and conver to mono signal
      audio_novocal = librosa.to_mono(np.transpose(stems[1] + stems[2] + stems[3]))
      audio_vocal = librosa.to_mono(np.transpose(stems[4]))

      # compute log mel spectrogram and convert to pytorch tensor
      logmelspec_novocal = torch.from_numpy(self._mel_spectrogram(audio_novocal, rate))
      logmelspec_vocal = torch.from_numpy(self._mel_spectrogram(audio_vocal, rate))

      start_ndx = 0

      for step in range(window_size // step_size):
        cropped_logmelspec_novocal = logmelspec_novocal[:, start_ndx:]
        cropped_logmelspec_vocal = logmelspec_vocal[:, start_ndx:]
        num_slices = cropped_logmelspec_novocal.shape[1] // window_size

        # chop off the last bit so that number of stft steps is a multiple of window_size
        cropped_logmelspec_novocal = cropped_logmelspec_novocal[: , 0:num_slices*window_size]
        cropped_logmelspec_vocal = cropped_logmelspec_vocal[:, 0:num_slices*window_size]

        # reshape tensors into chunks of size 128x(window_size)
        # first dimension is number of chunks
        cropped_logmelspec_novocal = torch.transpose(torch.reshape(cropped_logmelspec_novocal, (128, num_slices, window_size)), 0, 1)
        cropped_logmelspec_vocal = torch.transpose(torch.reshape(cropped_logmelspec_vocal, (128, num_slices, window_size)), 0, 1)

        # unsqueeze and concatenate these tensors. Then concatenate to the big tensor
        logmels = torch.cat((cropped_logmelspec_novocal.unsqueeze(1), cropped_logmelspec_vocal.unsqueeze(1)), 1)
        logmels = self.remove_silent_layers(logmels)
        self.mel_specs = torch.cat((self.mel_specs, logmels), 0)
        self.sample_rates = torch.cat((self.sample_rates, torch.full((num_slices,), rate)), 0)

        if num_songs % 10 == 0:
          print(str(num_songs) + " songs processed; produced " + str(self.mel_specs.shape[0]) + " spectrograms")

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

  def cat(self, other_ds):
    self.mel_specs = torch.cat((self.mel_specs, other_ds.mel_specs), 0)
    self.sample_rates = torch.cat((self.sample_rates, other_ds.sample_rates), 0)

  def remove_silent_layers(self, mel_specs, thresh=-30):
    '''Removes any spectrograms from mel_specs where the vocal track is too quiet.
    We define a chunk of audio to be 'too quiet' if the maximum value of a mel bin
    is below the threshold. '''
    nonzero_slices = []
    for ndx in range(mel_specs.shape[0]):
      if torch.max(mel_specs[ndx, 1, :, :]) >= thresh:
        nonzero_slices.append(ndx)

    return mel_specs[nonzero_slices]