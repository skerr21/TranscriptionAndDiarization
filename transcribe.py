import os
import wave
import json
import ffmpeg
from pyannote.audio import Pipeline
import torch
from faster_whisper import WhisperModel
from utils import overlap, stop_playback
import simpleaudio as sa
import tempfile


def transcribe_audio(audio_file):
    # Get the base file name
    base_file_name, extension = os.path.splitext(audio_file)
    transcription_file_name = f"{base_file_name}_transcription.json"

    # Check if a transcription already exists
    if os.path.exists(transcription_file_name):
        print(f"Transcription already exists for {audio_file}")
        return  # skip this file and go to the next one

    # Check if the file is in .wav format
    if extension.lower() != '.wav':
        # If not, convert it to .wav on-the-fly
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        stream = ffmpeg.input(audio_file)
        stream = ffmpeg.output(stream, temp_file.name)
        ffmpeg.run(stream, overwrite_output=True)
        print(f"Converted {audio_file} to .wav format.")
        audio_file = temp_file.name

    # Check if the converted file is readable as a .wav file
    try:
        wave.open(audio_file, 'rb')
        print(f"Successfully opened {audio_file} with wave module.")
    except wave.Error as e:
        print(f"Failed to open {audio_file} with wave module.")
        raise ValueError(f"Could not open {audio_file} as a .wav file: {e}")

    # Transcribe the audio file
    model = WhisperModel()
    transcription = model.transcribe(audio_file)

    # Save the transcription to a JSON file
    with open(transcription_file_name, "w") as f:
        json.dump(transcription, f)

    # Remove the temporary file if it was created
    if extension.lower() != '.wav':
        os.remove(audio_file)