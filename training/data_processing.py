import pandas as pd
import numpy as np

NUMBER_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_note(midi_number):
    if not (0 <= midi_number <= 127):
        return "Invalid MIDI number (must be between 0 and 127)"
    
    note_name = NUMBER_TO_NOTE[midi_number % 12]
    octave = (midi_number // 12) - 1
    
    return f"{note_name}{octave}"

def pick_pieces(pieces = ["Ballade No. 1 in G Minor, Op. 23"]):
    base_path = "training_data/maestro-v3.0.0/"
    df = pd.read_csv(f"{base_path}maestro-v3.0.0.csv")  

    filtered_data = df[df["canonical_title"].isin(pieces)]

    filtered_data = filtered_data["midi_filename"].tolist()


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
    import pretty_midi
    import numpy as np
    pm = pretty_midi.PrettyMIDI(midi_file)
    # Select an instrument (assumes that the desired soloist is in one track)
    instrument = pm.instruments[instrument_index]
    # Sort the notes by their start time
    notes = sorted(instrument.notes, key=lambda note: note.start)
    onset_times = np.array([note.start for note in notes])
    pitches = np.array([note.pitch for note in notes])
    if include_notes:
        return np.stack((pitches, onset_times))
    return onset_times

    
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
    tensor = prepare_tensor("assets/real_chopin.mid", "assets/reference_chopin.mid",)
    for i in range(len(tensor) - 1):
        if tensor[i + 1][2] - tensor[i][2] == 0: # 0 division error for the model
            print(f"anomoly found at point {i + 1}")
            print(tensor[i - 10: i + 10])
            break
