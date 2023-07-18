import os
import glob
import time
from pyannote.audio import Pipeline
from pydub import AudioSegment
from faster_whisper import WhisperModel
from transcribe import transcribe_audio

start_time = time.time()

# List all audio and video files in the directory
directory = 'F:/transcriber test'  # replace with your directory path
audio_files = glob.glob(os.path.join(directory, '*.wav')) + \
              glob.glob(os.path.join(directory, '*.mp3')) + \
              glob.glob(os.path.join(directory, '*.aac')) + \
              glob.glob(os.path.join(directory, '*.ogg')) + \
              glob.glob(os.path.join(directory, '*.flac')) + \
              glob.glob(os.path.join(directory, '*.m4a')) + \
              glob.glob(os.path.join(directory, '*.wma')) + \
              glob.glob(os.path.join(directory, '*.opus')) + \
              glob.glob(os.path.join(directory, '*.alac')) + \
              glob.glob(os.path.join(directory, '*.mp4')) + \
              glob.glob(os.path.join(directory, '*.avi')) + \
              glob.glob(os.path.join(directory, '*.mkv')) + \
              glob.glob(os.path.join(directory, '*.flv')) + \
              glob.glob(os.path.join(directory, '*.mov')) + \
              glob.glob(os.path.join(directory, '*.wmv')) + \
              glob.glob(os.path.join(directory, '*.webm'))

# Create a list of audio files that need to be transcribed
transcription_files = []
for audio_file in audio_files:
    base_file_name, extension = os.path.splitext(audio_file)
    transcription_file_name = f"{base_file_name}_transcription.json"
    if not os.path.exists(transcription_file_name):
        transcription_files.append(audio_file)

# Transcribe and diarize audio files
for audio_file in transcription_files:
    transcribe_audio(audio_file)

end_time = time.time()
execution_time = end_time - start_time
execution_time_minutes = execution_time / 60
print(f"Execution time: {execution_time_minutes} minutes")