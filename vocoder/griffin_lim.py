import numpy as np
import librosa
import librosa.display

def convert_to_audio(spectrograms, batch_size = 8, n_fft=2048, hop_length=512, power=2.0, n_iter=32):
    audio_files = []
    sr = 44100
    
    for i in range(0, len(mel_spectrograms), batch_size):
        batch = spectrograms[i:i+batch_size]
        batch_audio = []
        
        for mel_spec in batch:
            # Convert Mel spectrogram back to linear spectrogram
            linear_spec = librosa.feature.inverse.mel_to_stft(
                mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length, power=power
            )
            
            # Apply Griffin-Lim algorithm for phase reconstruction
            audio = librosa.griffinlim(
                linear_spec, hop_length=hop_length, n_iter=n_iter
            )
            
            batch_audio.append(audio)
        
        audio_files.extend(batch_audio)
        print(f"Processed batch {i//batch_size + 1} of {(len(mel_spectrograms) + batch_size - 1)//batch_size}")
    
    return audio_files