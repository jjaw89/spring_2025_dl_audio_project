# pip install transformers
import torch
from transformers import HifiGanModel
import numpy as np

def generate_audio_with_hifigan(mel_spectrograms: List, batch_size = 8)
    # this model is specifically trained to convert mel spectrograms
    # I think this model expects the 80 frequency bands, need to re-examine 
    hifigan = HifiGanModel.from_pretrained("speechbrain/hifi-gan-vocoder")

    for i in range(0, len(mel_spectrograms), batch_size):
        batch = mel_spectrograms[i:i+batch_size]
        tensor = torch.FloatTensor(np.stack(batch))
        
        with torch.no_grad():
            audio_batch = hifigan(tensor).squeeze().cpu().numpy()
        
        for j, audio in enumerate(audio_batch):
            sf.write(f"output/audio_{i+j}.wav", audio, 44100)
            
        print(f"Processed batch {i//batch_size + 1} of {(len(mel_spectrograms) + batch_size - 1)//batch_size}")