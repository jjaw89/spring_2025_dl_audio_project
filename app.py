import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import torch
import io
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
import torch.nn.functional as F
# add a path to Wave-U-Net
import sys
sys.path.append('Wave-U-Net-Pytorch')

import model.utils as model_utils
import utils
from model.waveunet import Waveunet


# load the generator
model_config_gen = {
    "num_inputs": 256,  # Two spectrograms concatenated (2 * 128 mel bins)
    "num_outputs": 128,
    "num_channels": [512*2, 512*4, 512*8],
    "instruments": ["vocal"],
    "kernel_size": 3,
    "target_output_size": 256,
    "conv_type": "normal",
    "res": "fixed",
    "separate": False,
    "depth": 1,
    "strides": 2
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Waveunet(**model_config_gen).to(device)
generator.eval()
# load parameters if exist
state_time = "20250421-033033"
model_dir  = "models"
if os.path.exists(f"{model_dir}/generator_state_dict_{state_time}.pt"):
    gen_state  = torch.load(f"{model_dir}/generator_state_dict_{state_time}.pt",         map_location=device)
    generator.load_state_dict(gen_state)


# Proprecess the audio for model inference
class AudioProcessor:
    def __init__(self, window_size=256, output_length=289, sr=44100):

        self.window_size = window_size
        self.output_length = output_length
        self.sr = sr

    def load_audio(self, file_data, sr=44100):

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(file_data)
            tmp_path = tmp_file.name
        
        # Load the audio file
        audio, sr = librosa.load(tmp_path, sr=sr)
        
        # Clean up
        os.unlink(tmp_path)
        
        return audio, sr
    
    def process_audio_files(self, audio_speech, audio_acc):

        # Compute log mel spectrograms
        logmelspec_speech = torch.from_numpy(self._mel_spectrogram(audio_speech, self.sr))
        logmelspec_acc = torch.from_numpy(self._mel_spectrogram(audio_acc, self.sr))
        
        print(f"Shape of speech melspec: {logmelspec_speech.shape}")
        print(f"Shape of accompaniment melspec: {logmelspec_acc.shape}")
        
        # Truncate to ensure both have the same length
        min_frames = min(logmelspec_speech.shape[1], logmelspec_acc.shape[1])
        logmelspec_speech = logmelspec_speech[:, :min_frames]
        logmelspec_acc = logmelspec_acc[:, :min_frames]
        
        # Truncate to make it divisible by window_size
        frames_to_keep = (min_frames // self.window_size) * self.window_size
        logmelspec_speech = logmelspec_speech[:, :frames_to_keep]
        logmelspec_acc = logmelspec_acc[:, :frames_to_keep]
        
        # Reshape into chunks
        num_chunks = frames_to_keep // self.window_size
        print(f"Number of full chunks: {num_chunks}")
        speech_reshaped = logmelspec_speech.reshape(128, num_chunks, self.window_size)
        acc_reshaped = logmelspec_acc.reshape(128, num_chunks, self.window_size)
        
        # Then transpose to (num_chunks, 128, window_size)
        speech_chunks = speech_reshaped.permute(1, 0, 2)
        acc_chunks = acc_reshaped.permute(1, 0, 2)
        
        # Create a list to store each processed chunk
        processed_chunks = []
        
        for i in range(num_chunks):
            speech_chunk = speech_chunks[i]  # (128, window_size)
            acc_chunk = acc_chunks[i]        # (128, window_size)
            
            # Pad each chunk to the model's expected input length
            speech_padded = self._pad_for_model(speech_chunk)  # (128, output_length)
            acc_padded = self._pad_for_model(acc_chunk)        # (128, output_length)
            
            # Combine and add to list with batch dimension
            chunk_input = torch.cat([speech_padded.unsqueeze(0), acc_padded.unsqueeze(0)], dim=1)
            print(f"chunk_input shape {chunk_input.shape}")
            processed_chunks.append(chunk_input)
        
        # Stack all chunks into a single batch
        # Result: (num_chunks, 2, 128, output_length)
        batched_chunks = torch.cat(processed_chunks, dim=0)
        
        return batched_chunks

    def _mel_spectrogram(self, audio, sr):

        return librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr))
    
    def _pad_for_model(self, spectrogram, output_length=289):

        current_len = spectrogram.size(1)
        delta = output_length - current_len
        
        if delta > 0:
            left_pad_len = (delta // 2) + (delta % 2)
            right_pad_len = delta // 2
            padded = F.pad(spectrogram, (left_pad_len, right_pad_len), "constant", 0)
            return padded
        else:
            return spectrogram
    
    def reconstruct_output(self, model_outputs, sr=44100, n_fft=2048, hop_length=512, win_length=2048, n_iter=32):

        # Move to CPU and convert to numpy
        model_outputs = model_outputs.detach().cpu().numpy()
        
        # Get number of chunks
        num_chunks = model_outputs.shape[0]
        
        # Initialize a list to store audio segments
        audio_segments = []
        
        # Process each chunk
        for i in range(num_chunks):
            # Get the current mel spectrogram
            mel_spec = model_outputs[i]  # Shape: (128, 257)
            
            # Convert from log scale to linear scale
            mel_spec_linear = librosa.db_to_power(mel_spec)
            
            # Convert from mel spectrogram to linear spectrogram
            linear_spec = librosa.feature.inverse.mel_to_stft(
                mel_spec_linear, 
                sr=sr, 
                n_fft=n_fft,
                power=2.0
            )
            
            # Reconstruct phase using Griffin-Lim algorithm
            audio = librosa.griffinlim(
                linear_spec,
                hop_length=hop_length,
                win_length=win_length,
                n_iter=n_iter
            )
            
            # Append to list
            audio_segments.append(audio)
        
        # Concatenate all segments
        full_audio = np.concatenate(audio_segments)
        
        # Normalize audio (optional)
        if np.max(np.abs(full_audio)) > 1.0:
            full_audio = full_audio / np.max(np.abs(full_audio))
        
        return full_audio
    
# Set page configuration
st.set_page_config(
    page_title="Vocal Synthesis App",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1e3a8a;
    }
    .upload-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .st-emotion-cache-16txtl3 h4 {
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎤 AI Vocal Synthesis")
st.markdown("Transform your speech into singing with our deep learning model!")

# Sidebar for model options
with st.sidebar:
    st.header("Model Settings")
    model_quality = st.select_slider(
        "Output Quality",
        options=["Low", "Medium", "High"],
        value="Medium"
    )
    
    pitch_correction = st.checkbox("Apply Pitch Correction", value=True)

# Create two columns for file uploads
col1, col2 = st.columns(2)

# Function to load audio and show waveform
def load_audio_and_show_waveform(uploaded_file, title):
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load the audio file
        audio, sr = librosa.load(tmp_path, sr=None)
        
        # Display waveform
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(np.linspace(0, len(audio)/sr, len(audio)), audio, color='#1e3a8a', linewidth=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} Waveform")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # Audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Clean up
        os.unlink(tmp_path)
        
        return audio, sr
    return None, None

# Main application flow
with col1:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.header("Step 1: Upload Speech Audio")
    st.markdown("Upload a clear recording of your speech.")
    speech_file = st.file_uploader("Choose speech audio file", type=["wav", "mp3"])
    
    speech_audio, speech_sr = None, None
    if speech_file:
        speech_audio, speech_sr = load_audio_and_show_waveform(speech_file, "Speech")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.header("Step 2: Upload Accompaniment")
    st.markdown("Upload the instrumental music accompaniment.")
    accomp_file = st.file_uploader("Choose accompaniment audio file", type=["wav", "mp3"])
    
    accomp_audio, accomp_sr = None, None
    if accomp_file:
        accomp_audio, accomp_sr = load_audio_and_show_waveform(accomp_file, "Accompaniment")
    st.markdown("</div>", unsafe_allow_html=True)

# Create a section for processing
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.header("Step 3: Generate Vocal Synthesis")

# Get model instance
model = generator

output_sr = 44100
# Process button
if st.button("Generate Singing Voice", disabled=(speech_audio is None or accomp_audio is None)):
    with st.spinner("Processing... This may take a moment."):
        # Prepare settings dictionary
        settings = {
            "quality": model_quality,
            "pitch_correction": pitch_correction,
            "sample_rate": speech_sr  # Assuming both have the same sample rate
        }
        audio_processor = AudioProcessor()
        logmels = audio_processor.process_audio_files(speech_audio, accomp_audio).to(device)
        # Process through the model
        with torch.no_grad():
            generated_logmels = generator(logmels)["vocal"]
        output_audio = audio_processor.reconstruct_output(generated_logmels)
        print(len(output_audio) / output_sr)
        if output_audio is not None:
            # Display the result
            st.success("Vocal synthesis complete!")
            
            # Display waveform
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(np.linspace(0, len(output_audio)/output_sr, len(output_audio)), 
                   output_audio, color='#16a34a', linewidth=0.6)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Synthesized Vocal Waveform")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            # Create a temporary buffer to save the audio
            buffer = io.BytesIO()
            sf.write(buffer, output_audio, output_sr, format='WAV')
            buffer.seek(0)
            
            # Audio player
            st.audio(buffer, format='audio/wav')
            
            # Download button
            st.download_button(
                label="Download Synthesized Vocal",
                data=buffer,
                file_name="synthesized_vocal.wav",
                mime="audio/wav"
            )
            
            # Combine with accompaniment
            st.markdown("### Combined with Accompaniment")
            if st.button("Generate Combined Track"):
                with st.spinner("Mixing vocals with accompaniment..."):
                    # Simple mixing (in reality, you might want better mixing algorithms)
                    # Ensure same length
                    min_length = min(len(output_audio), len(accomp_audio))
                    mixed = output_audio[:min_length] * 0.7 + accomp_audio[:min_length] * 0.5
                    
                    # Normalize
                    mixed = mixed / np.max(np.abs(mixed))
                    
                    # Create a buffer for the mixed audio
                    mix_buffer = io.BytesIO()
                    sf.write(mix_buffer, mixed, output_sr, format='WAV')
                    mix_buffer.seek(0)
                    
                    # Audio player for mixed
                    st.audio(mix_buffer, format='audio/wav')
                    
                    # Download button for mixed
                    st.download_button(
                        label="Download Mixed Track",
                        data=mix_buffer,
                        file_name="mixed_track.wav",
                        mime="audio/wav"
                    )
else:
    if speech_audio is None or accomp_audio is None:
        st.info("Please upload both speech and accompaniment audio files to continue.")

st.markdown("</div>", unsafe_allow_html=True)

# Footer with information
st.markdown("---")
st.markdown("""
### How It Works

This application uses a deep learning model trained to transform human speech into singing vocals that match the provided musical accompaniment.

1. **Speech Analysis**: Your speech audio is analyzed for phonetic content and vocal characteristics.
2. **Music Analysis**: The accompaniment is analyzed for musical features like tempo, key, and melody.
3. **Vocal Synthesis**: Our AI model generates singing vocals that preserve your voice characteristics while matching the musical elements.

For best results:
- Use clear speech recordings with minimal background noise
- Provide high-quality instrumental accompaniment
- Experiment with different settings to find the perfect sound

""")