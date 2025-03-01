import pandas as pd
import numpy as np

from utils.midi_utils import extract_midi_onsets_and_pitches


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
    tensor = prepare_tensor("assets/real_chopin.mid", "assets/reference_chopin.mid",)
    for i in range(len(tensor) - 1):
        if tensor[i + 1][2] - tensor[i][2] == 0: # 0 division error for the model
            print(f"anomoly found at point {i + 1}")
            print(tensor[i - 10: i + 10])
            break
