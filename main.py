import os
import glob
import wave
import time
import whisper
import torch
import json
import threading
import simpleaudio as sa
from pyannote.core import Timeline, Segment
from pyannote.audio import Pipeline
from pydub import AudioSegment


# Function to check if intervals overlap
def overlap(interval_a, interval_b):
    return max(0, min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0]))

def stop_playback(play_obj, duration):
    time.sleep(duration)
    play_obj.stop()


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

    # Skip temporary files
    if 'temp_clip.wav' in audio_file:
        continue

    # Rest of the code...

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
        print(f"Converted {audio_file} to .wav format.")

    # Check if the converted file is readable as a .wav file
    try:
        wave.open(audio_file, 'rb')
        print(f"Successfully opened {audio_file} with wave module.")
    except wave.Error as e:
        print(f"Failed to open {audio_file} with wave module.")
        raise ValueError(f"Could not open {audio_file} as a .wav file: {e}")

    torch.cuda.init()
    torch.cuda.empty_cache()
    print(torch.cuda.is_initialized())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load whisper model and transcribe audio
    model = whisper.load_model("medium", device=device)
    result = {
        "transcription": model.transcribe(audio_file),
    }

    # Save transcription to JSON file
    with open(f"{audio_file}_output.json", "w") as f:
        json.dump(result, f)

    # Load pretrained diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=os.environ["HF_TOKEN"])
    torch.cuda.empty_cache()
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

    identified_speakers = {}

    # Save combined results to new text file
    output_text_file_name = f"{base_file_name}_transcription.txt"
    with open(output_text_file_name, "w", encoding='utf-8') as file:
        for result in combined_results:
            # Cut the clip for the current speaker
            start_time_s = result['start']  # start time in seconds
            end_time_s = result['end']  # end time in seconds

            # Calculate the duration and trim it to 7 seconds if it's longer
            clip_duration_s = end_time_s - start_time_s
            if clip_duration_s > 7:
                clip_duration_s = 7  # adjust clip duration to get a 7 second clip

            # Convert start and end times to frame numbers
            with wave.open(audio_file, 'rb') as audio:
                frames_per_second = audio.getframerate()
                start_frame = int(start_time_s * frames_per_second)
                num_frames = int(clip_duration_s * frames_per_second)  # calculate the number of frames for 7 seconds
                audio.setpos(start_frame)
                frames = audio.readframes(num_frames)  # read only the number of frames for 7 seconds

            # Save the clip to a temporary file
            clip_file = "temp_clip.wav"
            with wave.open(clip_file, 'wb') as clip:
                clip.setnchannels(audio.getnchannels())
                clip.setsampwidth(audio.getsampwidth())
                clip.setframerate(frames_per_second)
                clip.writeframes(frames)

            # Play the clip
            wave_obj = sa.WaveObject.from_wave_file(clip_file)
            play_obj = wave_obj.play()

# Stop the playback after 7 seconds
            stop_thread = threading.Thread(target=stop_playback, args=(play_obj, 7))
            stop_thread.start()
            stop_thread.end()


            # Ask for the identification of the speaker, only if it has not been identified before
            if result['speaker'] not in identified_speakers:
                print(f"Text spoken by {result['speaker']}: {result['text']}")
                speaker_id = input(f"Who is {result['speaker']}? ")
                identified_speakers[result['speaker']] = speaker_id
                print(f"You identified {result['speaker']} as: {speaker_id}")

            # Use the identified name if available, otherwise use the original label
            speaker_name = identified_speakers.get(result['speaker'], result['speaker'])
            file.write(f"{speaker_name}: {result['text']}\n")

end_time = time.time()
execution_time = end_time - start_time
execution_time_minutes = execution_time / 60
print(f"Execution time: {execution_time_minutes} minutes")