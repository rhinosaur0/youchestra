import numpy as np
from utils.lin_reg import train_linear_regression, predict_tempo_with_linear_regression
import librosa

def pitch_distance(ref_pitch, perf_pitch):
    semitone_diff = abs(ref_pitch - perf_pitch)
    if semitone_diff == 0:
        return 0  # Exact match
    elif semitone_diff <= 2:
        return 0.5  # Small penalty for nearby notes
    else:
        return 3.0  # Large penalty for outliers (hallucinations)

def dtw_pitch_alignment_with_speed(pitch_history, pitch_reference, accompaniment_progress, pitch_threshold=12):

    predicted_speed = 1.0
    pitch_reference = [(round(x[0], 2), x[1][0]) for x in pitch_reference]
    pitch_history = [(round(x[0], 2), x[1] + 12) for x in pitch_history]
    # +12 because the online piano doesn't have the same range as the reference

    user_times = np.array([t for (t, p) in pitch_history])
    user_pitches = np.array([p for (t, p) in pitch_history])
    ref_times = np.array([t for (t, p) in pitch_reference])
    ref_pitches = np.array([p for (t, p) in pitch_reference])

    # speed_factor, wp, matrix = compute_speed_factor(np.stack([user_times, user_pitches]), np.stack([ref_times, ref_pitches]))
    # print(f'Speed factor: {speed_factor}')
    # print(f'Warping path: {wp}')
    # print(f'DTW matrix: {matrix}')

    I = len(user_pitches)
    J = len(ref_pitches)

    # If no data, return safe defaults
    if I == 0 or J == 0:
        return None, None, 0.0

    # Initialize DP cost matrix and cumulative alignment path
    D = np.full((I+1, J+1), np.inf)
    D[0, :] = 0.0  # Allow starting anywhere in the reference

    # Fill the cost matrix
    for i in range(1, I+1):
        for j in range(1, J+1):
            pitch_dist = pitch_distance(user_pitches[i-1], ref_pitches[j-1]) 
            # pitch_dist = abs(user_pitches[i-1] - ref_pitches[j-1])

            if pitch_dist > pitch_threshold:
                D[i, j] = D[i-1, j]
            else:
                D[i, j] = pitch_dist + min(
                    D[i-1, j],    # Deletion (user note skipped)
                    D[i, j-1],    # Insertion (reference note skipped)
                    D[i-1, j-1]   # Match or substitute
                )

    alignment_path = []
    alignment_path_pitches = []
    i, j = I, np.argmin(D[I, :])   # End anywhere in the reference
    # print(D)
    while i > 0 and j > 0:
        alignment_path.append((i - 1, int(j - 1)))
        alignment_path_pitches.append((user_pitches[i-1], ref_pitches[j-1]))
        pitch_dist = pitch_distance(user_pitches[i-1], ref_pitches[j-1]) 
        # pitch_dist = abs(user_pitches[i-1] - ref_pitches[j-1])
        
        # Backtracking logic: match, deletion, or insertion
        if i != 1:
            if D[i, j] == pitch_dist + D[i-1, j-1]:
                i -= 1
                j -= 1
            elif D[i, j] == pitch_dist + D[i-1, j]:
                i -= 1
            else:
                j -= 1
        else:
            if D[i, j] == pitch_dist + D[i, j - 1]:
                j -= 1
            elif D[i, j] == pitch_dist + D[i-1, j - 1]:
                i -= 1
                j -= 1
            else:
                i -= 1

    alignment_path.reverse()  # Start-to-end order
    alignment_path_pitches.reverse()

    # print(alignment_path_pitches)
    solo_input = np.array([[user_pitches[i], user_times[i], ref_pitches[j], ref_times[j]] for i, j in alignment_path])
    user_attributes = np.stack(([user_times[i] for i, j in alignment_path], [user_pitches[i] for i, j in alignment_path]), axis=1).T
    ref_attributes = np.stack(([ref_times[j] for i, j in alignment_path], [ref_pitches[j] for i, j in alignment_path]), axis=1).T
    print(solo_input)
    predicted_speed, wp, matrix = compute_speed_factor(user_attributes, ref_attributes)
    print(f'Speed factor: {predicted_speed}')


    # Get the reference time aligned to the last user note
    last_user_idx, last_ref_idx = alignment_path[-1]
    current_ref_time = ref_times[last_ref_idx]

    return current_ref_time, predicted_speed


def combined_distance(x, y, time_weight=1.0, pitch_weight=0.0):
    """
    Compute a weighted distance between two feature vectors x and y.
    
    Each feature vector is of the form:
        [time, pitch]
    The distance is defined as:
        distance = time_weight * |x_time - y_time| + pitch_weight * |x_pitch - y_pitch|
    """
    time_diff = np.abs(x[0] - y[0])
    pitch_diff = pitch_distance(x[1], y[1])
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
    matrix, wp = librosa.sequence.dtw(X=soloist_features, Y=ref_features,
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
    return speed_factor, wp, matrix

