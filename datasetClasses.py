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
        self.mel_specs = torch.cat((self.mel_specs, logmels), 0)
        self.sample_rates = torch.cat((self.sample_rates, torch.full((num_slices,), rate)), 0)

        if num_songs % 5 == 0:
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