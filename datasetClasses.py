import os
import stempeg
import musdb
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class MusdbDataset(Dataset):
  """A MusdbDataset is a pytorch Dataset for the MUSDB18 data.
  
  Attributes
  --------
  mel_specs : torch.Tensor of size (num_samples, 2, num_mel_bins, window_size)
    This tensor holds 2xnum_samples many log mel spectrograms with num_mel_bins mel bins and 
    window_size many time steps. One for the accompaniment and one for the vocal. 
  sample_rates : torch.Tensor of size (num_samples)
    contains the sample rates for each sample. This is typical 44100Hz.

  Methods
  -------
  cat(self, other_ds)
    concatenates self and other_ds
  """
  def __init__(self, musDB, window_size = 256, step_size = 128):
    """
    Creates an instance of MusdbDataset
    
    Arguments
    ---------
    musDB : musdb.DB
      the musdb database object read from the data
    window_size : int, optional
      the number of time steps in each spectrogram
    step_size : int, optional
      indicates how far we step to start the next spectrogram. So each piece of audio in the middle
      of a song is contained in roughly window_size/step_size many spectrograms.
    """
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

    # remove the all zeros slice from initializing
    self.mel_specs = self.mel_specs[1: , : , : , :]
    self.sample_rates = self.sample_rates[1:]

  def __len__(self):
    """returns number of samples"""
    return self.mel_specs.shape[0]

  def __getitem__(self, ndx):
    """returns tuple (mel spectrogram of accompaniment, mel spectrogram of vocal, rate)"""
    return self.mel_specs[ndx, 0], self.mel_specs[ndx, 1], self.sample_rates[ndx]

  def _mel_spectrogram(self, audio, rate):
    """compute the log-mel-spectrogram of the audio at the given sample rate"""
    return librosa.power_to_db(librosa.feature.melspectrogram(y = audio, sr = rate))

  def cat(self, other_ds):
    """Concatenate two instances of MusdbDataset

    Arguments
    ---------
    other_ds : MusdbDataset
      the second dataset to be concatenated with self
    """
    self.mel_specs = torch.cat((self.mel_specs, other_ds.mel_specs), 0)
    self.sample_rates = torch.cat((self.sample_rates, other_ds.sample_rates), 0)






class LibriSpeechDataset(Dataset):
  """A subclass of pytorch.Dataset to handle the Librispeech Data

  Attributes
  --------
  mel_specs : torch.Tensor of size (num_samples, num_mel_bins, window_size)
    This tensor holds num_samples many log mel spectrograms with num_mel_bins mel bins and 
    window_size many time steps. 
  sample_rates : torch.Tensor of size (num_samples)
    contains the sample rates for each sample. This is typical 44100Hz.

  Methods
  -------
  cat(self, other_ds)
    concatenates self and other_ds
  """
  def __init__(self, path, window_size = 256, step_size = 128, num_specs = 20000):
    """
    Initializes an instance of MusdbDataset
    
    Arguments
    ---------
    path : str
      Path to the LibriSpeech dataset in workspace
    window_size : int, optional
      the number of time steps in each spectrogram
    step_size : int, optional
      indicates how far we step to start the next spectrogram. So each piece of audio in the middle
      of a song is contained in roughly window_size/step_size many spectrograms.
    num_specs : int, optional
      Dataset creation stops after num_specs samples. The argument allows the user to only
      build as many samples as are in the Musdb dataset. The LibriSpeech dataset has much more data.
    """
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
    """Return number of samples in dataset"""
    return self.mel_specs.shape[0]

  def __getitem__(self, ndx):
    """Return ndx-th sample in dataset with sample rate"""
    return self.mel_specs[ndx], self.sample_rates[ndx]

  def _mel_spectrogram(self, audio, rate):
    """compute the log-mel-spectrogram of the audio at the given sample rate"""
    return librosa.power_to_db(librosa.feature.melspectrogram(y = audio, sr = rate))



class AccompanimentVocalData(Dataset):
  """A wrapper to handle padding samples from the MusdbDataset

  Attribubes
  ----------
  musdb : MusdbDataset
    the Dataset with accompaniment and vocal spectrograms
  output_length : int, optional
    the size of spectrograms the generator Wave-U-Net models expect
  """
  def __init__(self, musdb_dataset, output_length = 289):
    self.musdb = musdb_dataset
    self.out_len = output_length

  def __len__(self):
    return len(self.musdb)

  def __getitem__(self, ndx):
    """Pads spectrograms with zeros according to output_length"""
    acc, voc, _ = self.musdb[ndx]
    delta = self.out_len - acc.size(-1)

    if delta > 0:
      # Half the remainder goes to the front
      left_pad_len = (delta // 2) + (delta % 2)  # 17
      right_pad_len = delta // 2                # 16
      acc_pad = F.pad(acc, (left_pad_len, right_pad_len), "constant", 0)
      voc_pad = F.pad(voc, (left_pad_len, right_pad_len), "constant", 0)
    else:
      acc_pad = acc
      voc_pad = voc

    return {"acc_no_pad" : acc,
            "voc_no_pad" : voc,
            "acc_pad": acc_pad,
            "voc_pad" : voc_pad
            }


class SpeechData(Dataset):
  """A wrapper to handle padding samples from the LibriSpeechDataset

  Attribubes
  ----------
  musdb : MusdbDataset
    the Dataset with speech spectrograms
  output_length : int, optional
    the size of spectrograms the generator Wave-U-Net models expect
  """
  def __init__(self, librispeech_dataset, output_length=289):
    self.librispeech_dataset = librispeech_dataset
    self.output_length = output_length

  def __len__(self):
    return len(self.librispeech_dataset)

  def __getitem__(self, index):
    """Pads spectrograms with zeros according to output_length"""
    speech, _ = self.librispeech_dataset[index]
    # If speech has multiple slices, pick the first slice
    if speech.dim() == 3:
      speech = speech[0]  # shape: [128, 256]
    current_len = speech.size(-1)
    delta = self.output_length - current_len

    if delta > 0:
      left_pad_len = (delta // 2) + (delta % 2)
      right_pad_len = delta // 2
      speech_pad = F.pad(speech, (left_pad_len, right_pad_len), "constant", 0)
    else:
      speech_pad = speech
    return {"no_pad" : speech, "pad" : speech_pad}
