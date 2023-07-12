import os
import glob
import wave
import time
import whisper
import torch
import json
from pyannote.core import Timeline, Segment  
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Function to check if intervals overlap
def overlap(interval_a, interval_b):
    return max(0, min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0]))

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

for audio_file in audio_files:
    # Get the base file name
    base_file_name, extension = os.path.splitext(audio_file)
    transcription_file_name = f"{base_file_name}_transcription.json"

    # Check if a transcription already exists
    if os.path.exists(transcription_file_name):
        print(f"Transcription already exists for {audio_file}")
        continue  # skip this file and go to the next one

    # Check if the file is in .wav format
    if extension.lower() != '.wav':
        # If not, convert it to .wav
        audio = AudioSegment.from_file(audio_file)
        audio_file = base_file_name + '.wav'
        audio.export(audio_file, format='wav')

    # Check if the converted file is readable as a .wav file
    try:
        wave.open(audio_file, 'rb')
    except wave.Error as e:
        raise ValueError(f"Could not open {audio_file} as a .wav file: {e}")

    # Load whisper model and transcribe audio
    model = whisper.load_model("medium")
    result = {
        "transcription": model.transcribe(audio_file),
    }

    # Save transcription to JSON file
    with open(f"{audio_file}_output.json", "w") as f:
        json.dump(result, f)

    # Load pretrained diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=os.environ["HF_TOKEN"])
    pipeline.to(torch.device('cuda'))

    # Apply pipeline to audio file
    diarization = pipeline(audio_file)

    # Load transcription from JSON file
    with open(f"{audio_file}_output.json", "r") as file:
        output_data = json.load(file)

    transcription = output_data['transcription']['segments']

    # Store combined results
    combined_results = []

    # Associate text segments with speaker labels and merge consecutive segments from the same speaker
    prev_speaker = None
    prev_text = ""
    prev_start = 0
    prev_end = 0
    for seg in transcription:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if overlap([seg['start'], seg['end']], [turn.start, turn.end]):
                speaker = f"speaker_{speaker}"
                if prev_speaker is not None and speaker == prev_speaker:
                    prev_text += " " + seg['text']  # merge with previous segment
                    prev_end = seg['end']
                else:
                    if prev_speaker is not None:
                        combined_results.append({
                            "start": prev_start,
                            "end": prev_end,
                            "speaker": prev_speaker,
                            "text": prev_text.strip()
                        })
                    prev_speaker = speaker
                    prev_text = seg['text']
                    prev_start = seg['start']
                    prev_end = seg['end']
                break  # Once we found a matching turn for a segment, we can stop the inner loop

     # Save combined results to new JSON file
    base_file_name, _ = os.path.splitext(audio_file)
    output_file_name = f"{base_file_name}_transcription.json"
    with open(output_file_name, "w") as file:
        json.dump(combined_results, file)

    # Save combined results to new text file
    output_text_file_name = f"{base_file_name}_transcription.txt"
    with open(output_text_file_name, "w") as file:
        for result in combined_results:
            file.write(f"{result['speaker']}: {result['text']}\n")


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
