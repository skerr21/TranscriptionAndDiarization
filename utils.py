# utils.py
import time

def overlap(interval_a, interval_b):
    return max(0, min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0]))

def stop_playback(play_obj, duration):
    print(f"Sleeping for {duration} seconds...")
    time.sleep(duration)
    print("Stopping playback...")
    play_obj.stop()
    print("Playback stopped.")