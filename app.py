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

# Load the model (placeholder for your actual model loading code)
@st.cache_resource
def load_model():
    # In a real implementation, you would load your trained model here
    # For example: model = torch.load("path_to_your_model.pth")
    # Return a dummy model for this demo
    class DummyModel:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        def synthesize(self, speech, accompaniment, settings):
            # This is a placeholder for your actual model inference
            # In a real application, this would process the inputs through your model
            
            # For demo purposes, just returning the original speech with some processing
            # to simulate a singing voice
            
            # Make sure both audios have the same sample rate
            if speech is None or accompaniment is None:
                return None, None
                
            # Apply some basic transformations to simulate singing
            # In reality, this would be your model's output
            duration = min(len(speech), len(accompaniment))
            synthetic_vocal = speech[:duration] * 0.7 + np.sin(np.linspace(0, 100, duration)) * 0.05
            
            # Apply pitch correction if enabled
            if settings["pitch_correction"]:
                # This is a placeholder for pitch correction
                # In a real implementation, you would apply your pitch correction algorithm here
                pass
            
            return synthetic_vocal, settings["sample_rate"]

    return DummyModel()

# Get model instance
model = load_model()

# Process button
if st.button("Generate Singing Voice", disabled=(speech_audio is None or accomp_audio is None)):
    with st.spinner("Processing... This may take a moment."):
        # Prepare settings dictionary
        settings = {
            "quality": model_quality,
            "pitch_correction": pitch_correction,
            "sample_rate": speech_sr  # Assuming both have the same sample rate
        }
        
        # Process through the model
        output_audio, output_sr = model.synthesize(speech_audio, accomp_audio, settings)
        
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