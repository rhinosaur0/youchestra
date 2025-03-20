import numpy as np
import pandas as pd
from music21 import converter

NUMBER_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def parse_midi(midi_path):
    
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


def extract_midi_onsets_and_pitches(midi_file, include_notes = False, instrument_index=0):
    """
    Extract note onset times and pitches from a MIDI file.
    
    Args:
      midi_file (str): Path to the MIDI file.
      instrument_index (int): Which instrument track to use (default is 0).
    
    Returns:
      onset_times (np.array): Array of note onset times (in seconds).
      pitches (np.array): Array of corresponding MIDI pitch numbers.
    """
    import numpy as np
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_file)
    
    # Select an instrument (assumes that the desired soloist is in one track)
    instrument = pm.instruments[instrument_index]
    # Sort the notes by their start time
    notes = sorted(instrument.notes, key=lambda note: note.start)
    onset_times = np.array([note.start for note in notes])
    pitches = np.array([note.pitch for note in notes])

    if include_notes:
        final = np.stack((pitches, onset_times))
        return final
    
    return onset_times

def midi_to_note(midi_number):
    if not (0 <= midi_number <= 127):
        return "Invalid MIDI number (must be between 0 and 127)"
    
    note_name = NUMBER_TO_NOTE[midi_number % 12]
    octave = (midi_number // 12) - 1
    
    return f"{note_name}{octave}"

def extract_midi_onsets_and_pitches(midi_file, include_notes = False, instrument_index=0):
    """
    Extract note onset times and pitches from a MIDI file.
    
    Args:
      midi_file (str): Path to the MIDI file.
      instrument_index (int): Which instrument track to use (default is 0).
    
    Returns:
      onset_times (np.array): Array of note onset times (in seconds).
      pitches (np.array): Array of corresponding MIDI pitch numbers.
    """
    import numpy as np
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_file)
    
    # Select an instrument (assumes that the desired soloist is in one track)
    instrument = pm.instruments[instrument_index]
    # Sort the notes by their start time
    notes = sorted(instrument.notes, key=lambda note: note.start)
    onset_times = np.array([note.start for note in notes])

    onset_times[35:] -= onset_times[35]
    onset_times = onset_times[35:]

    pitches = np.array([note.pitch for note in notes])
    pitches = pitches[35:]
    if include_notes:
        final = np.stack((pitches, onset_times))
        return final
    
    origin = onset_times[288]
    for i in range(289, 544):
        onset_times[i] += onset_times[i] - origin
    onset_times[544:] += (onset_times[543] - origin) / 2
    
    return onset_times

def write_midi_from_timings(timings, notes, window_size, output_midi_file="output.mid", default_duration=0.3):
    """
    Given a sequence of predicted timing differences, compute cumulative onset times and write a MIDI file.
    Each note is assigned a constant pitch and fixed duration.
    """
    # Compute cumulative onset times: first note starts at time 0.
    import pretty_midi
    note_onsets = [0]
    for a, t in enumerate(timings):
        note_onsets.append(note_onsets[-1] + t)
    
    # Create a PrettyMIDI object and a piano instrument.
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    for onset, note in zip(note_onsets, notes[window_size - 1:]):
        start_time = onset
        end_time = onset + default_duration  # fixed note duration
        note = pretty_midi.Note(velocity=100, pitch=int(note), start=start_time, end=end_time)
        piano.notes.append(note)
    
    pm.instruments.append(piano)

    pm.write(output_midi_file)
    print(f"MIDI file written to {output_midi_file}")


def pick_pieces(pieces = ["Ballade No. 1 in G Minor, Op. 23"]):
    base_path = "training_data/maestro-v3.0.0/"
    df = pd.read_csv(f"{base_path}maestro-v3.0.0.csv")  

    filtered_data = df[df["canonical_title"].isin(pieces)]

    filtered_data = filtered_data["midi_filename"].tolist()


def prepare_tensor(live_midi, reference_midi, include_everything=False):
    live_tensor= extract_midi_onsets_and_pitches(live_midi, include_notes = True)
    reference_tensor= extract_midi_onsets_and_pitches(reference_midi)

    # The code below was used to check for alignment between the two tensors
    # for i, (a, b) in enumerate(zip(live_tensor[2200:], reference_tensor[2200:])):
    #     if not a[0] == b[0]:
    #         print(f'anomoly found at point {i + 1}')
    #     if i > 100:
    #         break
    #     print(a, b)
    final_tensor = np.vstack((live_tensor, reference_tensor))
    return final_tensor


if __name__ == "__main__":
    print(extract_midi_onsets_and_pitches("assets/reference_chopin.mid")[:, :30])


