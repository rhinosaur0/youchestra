from utils.midi_utils import parse_midi
from solo_tracker import SoloTracker
from playback_engine import AccompanimentPlayer
from conductor import Conductor
import threading
import time

def main():
    # Load MIDI files

    solo_midi_path = "assets/solo.mid"
    accomp_midi_path = "training_data/maestro-v3.0.0/2017/MIDI-Unprocessed_046_PIANO046_MID--AUDIO-split_07-06-17_Piano-e_2-02_wav--4.midi"
    default_sec_per_beat = 0.3
    

    # Parse the MIDI files
    solo_events, _ = parse_midi(solo_midi_path)


    barrier = threading.Barrier(3)

    accomp_player = AccompanimentPlayer()
    accomp_player.load_events(accomp_midi_path)

    # Initialize solo tracker (mic input)
    solo_tracker = SoloTracker()
    
    # Start listening to the soloist
    solo_tracker_thread = threading.Thread(target=solo_tracker.start_listening, args = (barrier,),daemon=True)
    solo_tracker_thread.start()
    
    # Initialize conductor to synchronize soloist and accompanist
    conductor = Conductor(solo_events, accomp_player, solo_tracker)


    try:
        # Start accompaniment playback thread
        playback_thread = threading.Thread(target=accomp_player.start_playback, args = (barrier,), daemon=True)
        playback_thread.start()

        # Start conductor logic
        print('starting conductor')
        conductor_thread = threading.Thread(target=conductor.start, args = (barrier, default_sec_per_beat), daemon=True)
        conductor_thread.start()

        # Wait for threads to complete or interrupt with Ctrl+C
        while playback_thread.is_alive() and conductor_thread.is_alive():
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        print('Cleaning up resources...')
        solo_tracker.stop_listening()
        accomp_player.stop_playback()

if __name__ == "__main__":
    main()

