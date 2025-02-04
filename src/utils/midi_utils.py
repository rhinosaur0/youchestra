def parse_midi(midi_path):
    from music21 import converter
    """
    Parse a MIDI file and return a list of note events.
    Each event is a tuple: (start_time, [pitch], duration)
    - start_time: When the note starts in seconds.
    - pitch: List of MIDI pitch values that play simultaneously.
    - duration: How long the notes play in seconds.
    """
    score = converter.parse(midi_path)
    
    # Flatten all parts and extract notes/rests
    list_parts = [part.flat.notesAndRests for part in score.parts]
    
    events = []  # To hold all events from all tracks
    default_sec_per_beat = 0.3  # Example tempo: 200 BPM

    for part in list_parts:
        current_time = 0.0
        track_events = []
        
        for i, n in enumerate(part):
            duration = n.duration.quarterLength * default_sec_per_beat
            if n.isNote:
                track_events.append((current_time, [n.pitch.midi], duration))
            elif n.isChord:
                pitches = [p.midi for p in n.pitches]
                track_events.append((current_time, pitches, duration))
            
            current_time += duration
            current_time = round(current_time, 8)
        
        events += track_events

    # Sort all events by start time
    events.sort(key=lambda x: x[0])

    # Combine simultaneous events
    combined_events = []
    current_time = None
    active_pitches = []
    
    for event in events:
        start_time, pitches, duration = event
        if current_time is None or start_time > current_time:
            # If a new time, finalize the last set of pitches
            if active_pitches:
                combined_events.append((current_time, active_pitches, last_duration))
            # Start a new event
            current_time = start_time
            active_pitches = pitches
            last_duration = duration
        else:
            # Combine pitches occurring at the same time
            active_pitches += pitches

    # Finalize the last event
    if active_pitches:
        combined_events.append((current_time, active_pitches, last_duration))

    return combined_events, default_sec_per_beat

def extract_midi_onsets_and_pitches(midi_file, instrument_index=0):
    """
    Extract note onset times and pitches from a MIDI file.
    
    Args:
      midi_file (str): Path to the MIDI file.
      instrument_index (int): Which instrument track to use (default is 0).
    
    Returns:
      onset_times (np.array): Array of note onset times (in seconds).
      pitches (np.array): Array of corresponding MIDI pitch numbers.
    """
    import pretty_midi
    import numpy as np
    pm = pretty_midi.PrettyMIDI(midi_file)
    # Select an instrument (assumes that the desired soloist is in one track)
    instrument = pm.instruments[instrument_index]
    # Sort the notes by their start time
    notes = sorted(instrument.notes, key=lambda note: note.start)
    onset_times = np.array([note.start for note in notes])
    pitches = np.array([note.pitch for note in notes])
    return np.stack((onset_times, pitches))

import time
from threading import Thread

class Note:
    def __init__(self, id):
        self.id = id
        self.playuntil = 0
        self.channel = 0
        self.velocity = 0
        self.particles = None
        self.is_on = False


if __name__ == "__main__":
    print(parse_midi("assets/solo.mid")[0][:10])
    print(extract_midi_onsets_and_pitches("assets/solo.mid")[:, :10])


