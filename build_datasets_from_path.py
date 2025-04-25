import stempeg
import musdb
from dataset_classes import MusdbDataset, LibriSpeechDataset

def create_musdb_dataset(path, window_size = 256, step_size = 128, subsets = "train"):
    """Returns a MusdbDataset object created from the songs found in path

    Arguments
    ---------
    path : str
        A string containing the path to the MUSDB datasets in the workspace
    subsets : str, optional
        A string to determine if the user wants train or test data
        This string must either be "train" or "test"
    window_size : int, optional
        the number of time steps in the log mel spectrogram. 256 steps is about
        2 seconds worth of audio
    step_size : int, optional
        After computing a spectrogram, the program takes a step_size in time.
        So each piece of audio in the song is contained in roughly 
        window_size // step_size many spectrograms
    """
    if subsets == "train" or  subsets == "test":
        musDB = musdb.DB(path, subsets=subsets)
        musdb_dataset = MusdbDataset(musDB, 
                                     window_size = window_size, 
                                     step_size = step_size)
    else:
        raise Exception("Invalid subsets argument, must be either train or test")
    
    return musdb_dataset


def create_librispeech_dataset(path, window_size = 256, step_size = 128):
    """Returns a LibriSpeechDataset object created from the audio found in path

    Arguments
    ---------
    path : str
        A string containing the path to the LibriSpeech dataset in the workspace
    window_size : int, optional
        the number of time steps in the log mel spectrogram. 256 steps is about
        2 seconds worth of audio
    step_size : int, optional
        After computing a spectrogram, the program takes a step_size in time.
        So each piece of audio in the song is contained in roughly 
        window_size // step_size many spectrograms
    """
    librispeech_dataset = LibriSpeechDataset(path, 
                                             window_size = window_size, 
                                             step_size = step_size)
    return librispeech_dataset
