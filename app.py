import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os
import torch
import numpy as np

# Page Config
st.set_page_config(page_title="AI Speech Hub", page_icon="🎙️")
st.title("🎙️ AI Speech Hub")
st.markdown("Convert Speech to Text or Text to Speech instantly.")

# Load Speech-to-Text Model (Whisper)
@st.cache_resource
def load_stt_model():
    # 'base' is fast and accurate for web apps
    return pipeline("automatic-speech-recognition", model="openai/whisper-base")

stt_pipe = load_stt_model()

# Select Mode
mode = st.radio("Choose a Tool:", ("Speech-to-Text (STT)", "Text-to-Speech (TTS)"))

# --- MODE 1: SPEECH TO TEXT ---
if mode == "Speech-to-Text (STT)":
    st.header("🎤 Voice to Text")
    audio_file = st.audio_input("Record your voice")
    
    if audio_file:
        with st.spinner("Transcribing..."):
            # Process the recorded audio
            text_output = stt_pipe(audio_file.read())["text"]
            st.success("### Transcribed Text:")
            st.write(text_output)
            st.button("Clear")

# --- MODE 2: TEXT TO SPEECH ---
else:
    st.header("🔊 Text to Voice")
    user_text = st.text_area("Enter text you want the AI to speak:", placeholder="Hello! How are you today?")
    
    if st.button("Convert to Speech"):
        if user_text.strip() != "":
            with st.spinner("Generating Audio..."):
                # Use gTTS to create audio
                tts = gTTS(text=user_text, lang='en')
                tts.save("temp_audio.mp3")
                
                # Play the audio in Streamlit
                st.audio("temp_audio.mp3", format="audio/mp3")
                
                # Cleanup (optional)
                os.remove("temp_audio.mp3")
        else:
            st.warning("Please enter some text first!")
