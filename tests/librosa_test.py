import numpy as np
import pretty_midi
import librosa
import librosa.display
import matplotlib.pyplot as plt

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
    pm = pretty_midi.PrettyMIDI(midi_file)
    # Select an instrument (assumes that the desired soloist is in one track)
    instrument = pm.instruments[instrument_index]
    # Sort the notes by their start time
    notes = sorted(instrument.notes, key=lambda note: note.start)
    onset_times = np.array([note.start for note in notes])
    pitches = np.array([note.pitch for note in notes])
    return onset_times[:20], pitches[:20]

def combined_distance(x, y, time_weight=1.0, pitch_weight=0.5):
    """
    Compute a weighted distance between two feature vectors x and y.
    
    Each feature vector is of the form:
        [time, pitch]
    The distance is defined as:
        distance = time_weight * |x_time - y_time| + pitch_weight * |x_pitch - y_pitch|
    """
    time_diff = np.abs(x[0] - y[0])
    pitch_diff = np.abs(x[1] - y[1])
    return time_weight * time_diff + pitch_weight * pitch_diff

def compute_speed_factor(soloist_features, ref_features, time_weight=1.0, pitch_weight=0.5):
    """
    Compute a speed factor by aligning the soloist and reference sequences (each as [time, pitch])
    using DTW. The speed factor is estimated as the median ratio between consecutive intervals
    in the soloist performance and the reference.
    
    Args:
      soloist_features (np.array): A 2 x N array where the first row is onset times and
                                   the second row is pitches.
      ref_features (np.array): A 2 x M array with the reference onset times and pitches.
      time_weight (float): Weight for time differences.
      pitch_weight (float): Weight for pitch differences.
    
    Returns:
      speed_factor (float): The median ratio of soloist interval to reference interval.
      wp (np.array): The DTW warping path (an array of index pairs).
    """
    # Run DTW using our custom metric.
    D, wp = librosa.sequence.dtw(X=soloist_features, Y=ref_features,
                                  metric=lambda x, y: combined_distance(x, y, time_weight, pitch_weight))
    # The warping path is returned in reverse order; reverse it to be chronological.
    wp = np.array(wp)[::-1]
    
    # Compute local interval ratios along the warping path.
    ratios = []
    soloist_times = soloist_features[0]
    ref_times = ref_features[0]
    for i in range(1, len(wp)):
        idx_solo_prev, idx_ref_prev = wp[i-1]
        idx_solo, idx_ref = wp[i]
        delta_solo = soloist_times[idx_solo] - soloist_times[idx_solo_prev]
        delta_ref = ref_times[idx_ref] - ref_times[idx_ref_prev]
        if delta_ref > 0:
            ratios.append(delta_solo / delta_ref)
    speed_factor = np.median(ratios) if ratios else 1.0
    return speed_factor, wp

# Example usage:
if __name__ == "__main__":
    # Path to your soloist MIDI file.
    midi_file = "assets/solo.mid"  # Replace with your actual MIDI file path.
    
    # Extract the soloist's onset times and pitches from the MIDI file.
    soloist_onset_times, soloist_pitches = extract_midi_onsets_and_pitches(midi_file, instrument_index=0)
    print("Soloist onset times:", soloist_onset_times)
    print("Soloist pitches:", soloist_pitches)
    
    # Create the soloist feature matrix: each note as [time, pitch].
    soloist_features = np.vstack((soloist_onset_times, soloist_pitches))
    
    # Create a reference sequence. For example, suppose you have a metronomic reference:
    # Let’s assume a note every 0.5 seconds over a fixed duration.
    ref_onset_times = soloist_onset_times * 10  # Adjust duration as needed.
    # And assume a reference pitch pattern (this could be constant or a predefined sequence).
    # For example, a simple ascending scale:
    ref_pitches = soloist_pitches  # MIDI pitches from 60 to 72.
    ref_features = np.vstack((ref_onset_times, ref_pitches))



    
    # Use DTW (with both timing and pitch considerations) to compute a speed factor.
    speed_factor, warping_path = compute_speed_factor(soloist_features, ref_features,
                                                      time_weight=1.0, pitch_weight=0.5)
    print(f"Estimated speed factor (considering both time and pitch): {speed_factor:.2f}")
    
    # Optional: visualize the DTW cost matrix and warping path.
    D, _ = librosa.sequence.dtw(X=soloist_features, Y=ref_features,
                                  metric=lambda x, y: combined_distance(x, y, 1.0, 0.5))
    plt.figure(figsize=(8, 6))
    librosa.display.specshow(D, x_axis='frames', y_axis='frames')
    plt.plot([p[1] for p in warping_path], [p[0] for p in warping_path],
             marker='o', color='r', label='Warping Path')
    plt.title("DTW Cost Matrix and Warping Path")
    plt.xlabel("Reference Index")
    plt.ylabel("Soloist Index")
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()
