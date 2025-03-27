from utils.midi_utils import parse_midi
from threads.solo_tracker import SoloTracker
from threads.playback_engine import AccompanimentPlayer
from threads.conductor import Conductor
import threading
import time
import argparse

def main(args):

    solo_midi_path = args.solo_midi_path
    accomp_midi_path = args.accomp_midi_path
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

        while playback_thread.is_alive() and conductor_thread.is_alive() and not solo_tracker.plot:
            time.sleep(0.05)

        solo_tracker.plot_debug_data()


    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        print('Cleaning up resources...')
        solo_tracker.stop_listening()
        accomp_player.stop_playback()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-s', '--solo_midi_path', type=str, help='path to solo midi file', default='assets/solo.mid')
    parser.add_argument('-a', '--accomp_midi_path', type=str, help='path to accompaniment midi file', default='assets/accompaniment.mid')
    main(parser.parse_args())

