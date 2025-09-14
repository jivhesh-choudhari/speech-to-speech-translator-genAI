# app.py
import streamlit as st
import tempfile
import os
import numpy as np
import wave
from audio_recorder_streamlit import audio_recorder
import uuid
# backend
from dotenv import(load_dotenv)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from openai import OpenAI
load_dotenv()


#=====================#
# Application Backend #
#=====================#

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) or st.secrets["OPENAI_API_KEY"]

class Translation(BaseModel):
    translation:str = Field(...,description='The result translation in a string format')


# --- 1. Transcription (STT) ---
def transcribe_audio(file_path: str) -> str:
    """
    Transcribes audio using OpenAI API.
    """
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
        )
    return transcript.text


# --- 2. Translation (TTT) ---
def translate_text(text: str, target_language: str) -> str:
    """
    Uses GPT model to translate text into target language.
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = PydanticOutputParser(pydantic_object=Translation)
    template = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are a translation engine. Translate the input text to given language only. Output only the translated text, no extra commentary.'),
            ('human', 'Translate the following text into {language}:{text} \n\n {format_instructions}')
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    chain = template | model | parser
    result = chain.invoke({"text": text, "language": target_language})
    return result.translation


# --- 3. Speech Generation (TTS) ---
def generate_speech(text: str, out_path="out.wav"):
    """
    Generate speech from text using OpenAI TTS.
    """
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )

    # Stream bytes to file
    with open(out_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)

    return out_path

# --- 4. Optimiation ---
def is_speech_present(wav_path: str, threshold: float = 500) -> bool:
    """
    Check if a WAV file likely contains human speech.
    threshold: RMS amplitude threshold; increase for noisier environments
    Returns True if speech is likely present.
    """
    with wave.open(wav_path, 'rb') as wf:
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

    # Convert bytes to numpy array
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    audio_data = np.frombuffer(frames, dtype=dtype)
    if n_channels > 1:
        audio_data = audio_data.reshape(-1, n_channels)
        audio_data = audio_data.mean(axis=1)  # convert to mono

    rms = np.sqrt(np.mean(audio_data**2))

    return rms > threshold

#======================#
# Application GUI code #
#======================#

st.title("Voice Translator 🎙️")

st.write("Record your speech and translate it!")

LANGUAGE_MAP = {
    "English": "eng",
    "French": "fra",
    "Hindi": "hin",
    "German": "deu",
    "Spanish": "spa",
    "Italian": "ita"
}

# audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
target_lang = st.selectbox("Select output language", list(LANGUAGE_MAP.keys()))

audio_bytes = audio_recorder(
    text="Click to record", 
    recording_color="#e74c3c", 
    neutral_color="#2ecc71", 
    icon_name="microphone",
)


if audio_bytes:

    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
    with open(tmp_path,'wb') as tmp_file:
        tmp_file.write(audio_bytes)
    tmp_file.close()

    if not is_speech_present(tmp_path):
        st.warning('No Speech Detected')
    else:
        iso_code = LANGUAGE_MAP[target_lang]

        source_text = transcribe_audio(tmp_path)

        translated_text = translate_text(source_text, iso_code)
        audio_path = generate_speech(translated_text)

        st.subheader("Transcription")
        st.write(source_text)
        st.subheader("Translation")
        st.write(translated_text)
        st.audio(audio_path)

    os.remove(tmp_path)
