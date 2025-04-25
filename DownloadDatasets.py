"""
These functions download the Musdb and Librispeech datasets. The user must input the file path
to the desired destination for the data.
"""
DATA_DEST_PATH = " PUT PATH HERE "

import requests
import zipfile
import tarfile


def download_musdb_data(data_dest_path):
    """
    Download the musdb dataset
    data_dest_path : str
        path to the desired location of the data in users file system
    """
    musdb_zip_file_name = data_dest_path + "/musdb18.zip"
    musdb_url = "https://zenodo.org/records/1117372/files/musdb18.zip"

    # Download the musdb zip file
    r = requests.get(musdb_url, stream = True)
    with open(musdb_zip_file_name, "wb") as file:
        for block in r.iter_content(chunk_size = 1024):
            if block:
                file.write(block)

    # Unzip
    with zipfile.ZipFile(musdb_zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(musdb_dest_path)


def download_librispeech_data(data_dest_path):
    """
    Download the librispeech datasets
    data_dest_path : str
        path to the desired location of the data in users file system
    """
    librispeech_train_tar_file_name = data_dest_path + "/train-clean-100.tar.gz"
    librispeech_test_tar_file_name = data_dest_path + "/test-clean.tar.gz"
    librispeech_train_url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    librispeech_test_url = "https://www.openslr.org/resources/12/test-clean.tar.gz"

    # Download Librispeech training data tar file
    r = requests.get(librispeech_train_url, stream = True)
    with open(librispeech_train_tar_file_name, "wb") as file:
        for block in r.iter_content(chunk_size = 1024):
            if block:
                file.write(block)

    # Download Librispeech test data tar file
    r = requests.get(librispeech_test_url, stream = True)
    with open(librispeech_test_tar_file_name, "wb") as file:
        for block in r.iter_content(chunk_size = 1024):
            if block:
                file.write(block)

    # extract train data tar file
    with tarfile.open(librispeech_train_tar_file_name) as tarobj:
        tarobj.extractall(librispeech_dest_path)

    # extract test data tar file
    with tarfile.open(librispeech_test_tar_file_name) as tarobj:
        tarobj.extractall(librispeech_dest_path)
