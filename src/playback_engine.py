import time
import mido
import os
from threading import Thread, Barrier
from utils.midi_utils import Note

class AccompanimentPlayer:
    def __init__(self):
        self.midi_out = mido.open_output()
        self.mid = None
        self.tempo_factor = 1.0
        self.playing = False
        self.partition = None
        self.notes = []
        self.current_progression = 0

    def load_events(self, midifile):

        for n in range(0, 88):
            self.notes.append(Note(n))

        try:
            mid = mido.MidiFile(midifile)
            filename, _ = os.path.splitext(os.path.split(midifile)[1])
            name = filename if mid.tracks[0].name == "" else mid.tracks[0].name
        except (OSError, EOFError) as e:
            pass
        partition, length = get_partition(mid)
        
        self.partition = partition

    def start_playback(self, barrier):
        self.playing = True
        barrier.wait()
        self._play_thread()

    def stop_playback(self):
        self.playing = False

    def _play_thread(self):
        i = 0
        modif = 0
        tstart = time.time()
        paused = False
        while self.partition[i]:
            tnow = time.time()
            if tnow + modif > self.partition[i]["time"] + tstart:
                if self.partition[i]["msg"].type == "note_on" and not self.partition[i]["note_off"]:
                    self.notes[self.partition[i]["msg"].note - 21].playuntil = (
                        tnow + self.partition[i]["new_velocity"] / 10
                    )
                    self.notes[self.partition[i]["msg"].note - 21].velocity = self.partition[i]["new_velocity"]
                    self.notes[self.partition[i]["msg"].note - 21].channel = self.partition[i]["msg"].channel
                self.midi_out.send(self.partition[i]["msg"])
                i += 1
                continue
            wait_time = self.partition[i]["time"] + tstart - (tnow + modif)
            if wait_time > 0.01:
                time.sleep(0.01)
            if paused:
                modif -= time.time() - tnow

    def adjust_tempo(self, factor):
        self.tempo_factor = max(0.8, min(2.0, factor))  # Limit tempo to 0.5x - 2.0x

    def retrieve_progression(self):
        return 0
    

def get_partition(mid):
    _time = 1
    partition = []
    for msg in mid:
        _time += msg.time
        if isinstance(msg, mido.MetaMessage):
            continue
        partition.append(
            {"time": _time, "msg": msg, "new_velocity": 0, "note_off": False}
        )
    partition.append(None)
    length = partition[-2]["time"]
 # print all channel from 0
    return partition, length

if __name__ == "__main__":
    player = AccompanimentPlayer()
    barrier = Barrier(1)
    player.load_events("assets/accompaniment.mid")


    playback_thread = Thread(target=player.start_playback, args=(barrier,))

    playback_thread.start()
    player.stop_playback()
    print("Playback stopped")


