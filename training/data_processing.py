import pandas as pd
import mido
import numpy as np

NUMBER_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_note(midi_number):
    if not (0 <= midi_number <= 127):
        return "Invalid MIDI number (must be between 0 and 127)"
    
    note_name = NUMBER_TO_NOTE[midi_number % 12]
    octave = (midi_number // 12) - 1
    
    return f"{note_name}{octave}"

def extract_notes_from_midi(file_path, include_notes=False):
    """Extract note and timing information from a MIDI file."""
    midi = mido.MidiFile(file_path)
    if include_notes:
        notes = np.empty((0, 2), dtype=object)  
    else:
        notes = np.empty((0, 1), dtype=object)
    current_time = 0  # To track absolute timing
    types = set()

    for track in midi.tracks:
        for msg in track:
            types.add(msg.type)
            if msg.type in ['control_change', 'program_change', 'set_tempo', 'end_of_track', 'note_on', 'time_signature', 'note_off']:  # We're only interested in note events
                current_time += msg.time / 480  # Update timing (delta time format)
                if msg.type == "note_on" and msg.velocity > 0:  # Ignore note-off and zero-velocity note-ons
                    # new_note = np.array([[midi_to_note(int(msg.note)), current_time]], dtype=object)
                    if include_notes:
                        # new_note = np.array([[msg, midi_to_note(int(msg.note)), current_time]], dtype=object)
                        new_note = np.array([[int(msg.note), current_time]], dtype=object)
                    else:
                        new_note = np.array([[current_time]], dtype=object)
                    notes = np.append(notes, new_note, axis=0)

    # print(types)
    return notes  # Already a numpy array

def pick_pieces(pieces = ["Ballade No. 1 in G Minor, Op. 23"]):
    base_path = "training_data/maestro-v3.0.0/"
    df = pd.read_csv(f"{base_path}maestro-v3.0.0.csv")  

    filtered_data = df[df["canonical_title"].isin(pieces)]

    filtered_data = filtered_data["midi_filename"].tolist()
    first_tensor = extract_notes_from_midi(f"{base_path}{filtered_data[0]}")

    
def prepare_tensor(live_midi, reference_midi):
    live_tensor= extract_notes_from_midi(live_midi, include_notes=True)
    reference_tensor= extract_notes_from_midi(reference_midi)

    # The code below was used to check for alignment between the two tensors
    # for i, (a, b) in enumerate(zip(live_tensor[2200:], reference_tensor[2200:])):
    #     if not a[0] == b[0]:
    #         print(f'anomoly found at point {i + 1}')
    #     if i > 100:
    #         break
    #     print(a, b)

    final_tensor = np.concatenate((live_tensor, reference_tensor), axis=1)
    return final_tensor





if __name__ == "__main__":
    tensor = prepare_tensor("assets/real_chopin.mid", "assets/reference_chopin.mid")
    print(tensor[100:150, :])
    for i in range(len(tensor) - 1):
        if tensor[i + 1][2] - tensor[i][2] == 0: # 0 division error for the model
            print(f"anomoly found at point {i + 1}")
            print(tensor[i - 10: i + 10])
            break
