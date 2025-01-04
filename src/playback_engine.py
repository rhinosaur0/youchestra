import rtmidi
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
import time

class AccompanimentPlayer:
    def __init__(self):
        self.midi_out = rtmidi.MidiOut()
        available_ports = self.midi_out.get_ports()
        if available_ports:
            for i, port in enumerate(available_ports):
                if "IAC Driver" in port:
                    print(f"Using MIDI Port: {port}")
                    self.midi_out.open_port(i)
                    break
            else:
                raise RuntimeError("IAC Driver not found in available MIDI ports.")
        else:
            raise RuntimeError("No MIDI output ports available.")

        self.events = []
        self.tempo_factor = 1.0
        self.playing = False

    def load_events(self, events):
        self.events = events

    def start_playback(self, barrier):
        
        self.playing = True
        barrier.wait()
        self._play_thread()

    def stop_playback(self):
        self.playing = False

    def _play_thread(self):
        start_time = time.time()
        for start_sec, pitch, dur_sec in self.events:
            self.current_progression = start_sec
            if not self.playing:
                break
            adj_start = start_sec / self.tempo_factor
            adj_dur = dur_sec / self.tempo_factor

            while time.time() - start_time < adj_start:
                if not self.playing:
                    return
                time.sleep(0.001)
            
            for i in range(0, len(pitch)):
                self.midi_out.send_message([NOTE_ON, pitch[i], 100])
            
            time.sleep(adj_dur)
            for i in range(0, len(pitch)):
                self.midi_out.send_message([NOTE_OFF, pitch[i], 0])
            

    def adjust_tempo(self, factor):
        self.tempo_factor = max(0.8, min(2.0, factor))  # Limit tempo to 0.5x - 2.0x

    def retrieve_progression(self):
        return self.current_progression
