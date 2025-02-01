import numpy as np
from utils.lin_reg import train_linear_regression, predict_tempo_with_linear_regression

def pitch_distance(ref_pitch, perf_pitch):
    semitone_diff = abs(ref_pitch - perf_pitch)
    if semitone_diff == 0:
        return 0  # Exact match
    elif semitone_diff <= 2:
        return 0.5  # Small penalty for nearby notes
    else:
        return 3.0  # Large penalty for outliers (hallucinations)

def dtw_pitch_alignment_with_speed(pitch_history, pitch_reference, accompaniment_progress, pitch_threshold=12):
    import numpy as np
    predicted_speed = 1.0
    pitch_reference = [(round(x[0], 2), x[1][0]) for x in pitch_reference]
    pitch_history = [(round(x[0], 2), x[1] + 12) for x in pitch_history]
    # +12 because the online piano doesn't have the same range as the reference

    user_times = np.array([t for (t, p) in pitch_history])
    user_pitches = np.array([p for (t, p) in pitch_history])
    ref_times = np.array([t for (t, p) in pitch_reference])
    ref_pitches = np.array([p for (t, p) in pitch_reference])

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
    
    while i > 0 and j > 0:
        alignment_path.append((i - 1, int(j - 1)))
        alignment_path_pitches.append((user_pitches[i-1], ref_pitches[j-1]))
        pitch_dist = pitch_distance(user_pitches[i-1], ref_pitches[j-1]) 
        
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

    model = train_linear_regression(alignment_path)
    predicted_speed = predict_tempo_with_linear_regression(model, alignment_path, user_times, ref_times)

    # Get the reference time aligned to the last user note
    last_user_idx, last_ref_idx = alignment_path[-1]
    current_ref_time = ref_times[last_ref_idx]

    return current_ref_time, predicted_speed


