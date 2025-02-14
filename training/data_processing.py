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

def extract_notes_from_midi(file_path):
    """Extract note and timing information from a MIDI file."""
    midi = mido.MidiFile(file_path)
    notes = np.empty((0, 2), dtype=object)  # Initialize an empty numpy array with object dtype for mixed types
    untouched_notes = []
    current_time = 0  # To track absolute timing
    old_time = 0

    for track in midi.tracks:
        for msg in track:
            untouched_notes.append(msg)
            if msg.type in ["note_on", "note_off"]:  # We're only interested in note events
                current_time += msg.time / 480  # Update timing (delta time format)

                if msg.type == "note_on" and msg.velocity > 0:  # Ignore note-off and zero-velocity note-ons
                    new_note = np.array([[midi_to_note(int(msg.note)), current_time]], dtype=object)
                    notes = np.append(notes, new_note, axis=0)
                    old_time = current_time

    return notes, untouched_notes  # Already a numpy array

def pick_pieces(pieces = ["Ballade No. 1 in G Minor, Op. 23"]):
    base_path = "training_data/maestro-v3.0.0/"
    df = pd.read_csv(f"{base_path}maestro-v3.0.0.csv")  # Replace with your actual CSV file path

    # Define the search condition (e.g., based on file name)

    # Filter rows where "Title" is in the list
    filtered_data = df[df["canonical_title"].isin(pieces)]

    filtered_data = filtered_data["midi_filename"].tolist()
    print(filtered_data[0])
    first_tensor = extract_notes_from_midi(f"{base_path}{filtered_data[0]}")
    print(first_tensor[:10, :])
    
def prepare_tensor(live_midi, reference_midi):
    live_tensor, untouched_live = extract_notes_from_midi(live_midi)
    reference_tensor, untouched_reference = extract_notes_from_midi(reference_midi)

    for i, (a, b) in enumerate(zip(live_tensor[2200:], reference_tensor[2200:])):
        if not a[0] == b[0]:
            print(f'anomoly found at point {i + 1}')
        if i > 100:
            break
        print(a, b)

    # print(live_tensor.shape)
    # print(reference_tensor.shape)

    # final_tensor = np.concatenate((live_tensor, reference_tensor), axis=1)
    # print(live_tensor[:10, :])
    # print(reference_tensor[100:200, :])





if __name__ == "__main__":
    prepare_tensor("assets/real_chopin.mid", "assets/reference_chopin.mid")