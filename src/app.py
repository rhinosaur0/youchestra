from utils.midi_utils import parse_midi
from solo_tracker import SoloTracker
from playback_engine import AccompanimentPlayer
from conductor import Conductor
import threading
import time

def main():

    solo_midi_path = "assets/solo.mid"
    accomp_midi_path = "assets/accompaniment.mid"
    default_sec_per_beat = 0.3
    

    solo_events, _ = parse_midi(solo_midi_path)

    barrier = threading.Barrier(3)

    accomp_player = AccompanimentPlayer()
    accomp_player.load_events(accomp_midi_path)

    solo_tracker = SoloTracker()
    solo_tracker_thread = threading.Thread(target=solo_tracker.start_listening, args = (barrier,),daemon=True)
    solo_tracker_thread.start()
    
    conductor = Conductor(solo_events, accomp_player, solo_tracker)

    playback_thread = threading.Thread(target=accomp_player.start_playback, args = (barrier,), daemon=True)
    playback_thread.start()

    try:
        print('starting conductor')
        conductor_thread = threading.Thread(target=conductor.start, args = (barrier, default_sec_per_beat), daemon=True)
        conductor_thread.start()

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

