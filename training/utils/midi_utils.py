

NUMBER_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

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
    import mido
    pm = pretty_midi.PrettyMIDI(midi_file)
    # midi_file = mido.MidiFile('../assets/reference_chopin.mid')
    # for msg in midi_file:
    #     if msg.type == 'note_on':
    #         print(msg)

    if midi_file == "../assets/reference_chopin.mid":
        beats = pm.get_downbeats()
    
    # Select an instrument (assumes that the desired soloist is in one track)
    instrument = pm.instruments[instrument_index]
    # Sort the notes by their start time
    notes = sorted(instrument.notes, key=lambda note: note.start)
    onset_times = np.array([note.start for note in notes])
    pitches = np.array([note.pitch for note in notes])
    if include_notes:
        return np.stack((pitches, onset_times))
    return onset_times

def write_midi_from_timings(timings, notes, window_size, output_midi_file="output.mid", default_duration=0.3):
    """
    Given a sequence of predicted timing differences, compute cumulative onset times and write a MIDI file.
    Each note is assigned a constant pitch and fixed duration.
    """
    # Compute cumulative onset times: first note starts at time 0.
    import pretty_midi
    note_onsets = [0]
    
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