import numpy as np
from utils.lin_reg import train_linear_regression, predict_tempo_with_linear_regression


def dtw_pitch_alignment_with_speed(pitch_history, pitch_reference, accompaniment_progress, pitch_threshold=2.0):
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
            pitch_dist = abs(user_pitches[i-1] - ref_pitches[j-1])

            D[i, j] = pitch_dist + min(
                D[i-1, j],    # Deletion (user note skipped)
                D[i, j-1],    # Insertion (reference note skipped)
                D[i-1, j-1]   # Match or substitute
            )

    # Backtrack to find the alignment path
    alignment_path = []
    i, j = I, np.argmin(D[I, :])  # End anywhere in the reference
    while i > 0 and j > 0:
        alignment_path.append((i-1, j-1))
        pitch_dist = abs(user_pitches[i-1] - ref_pitches[j-1])

        # Backtracking logic: match, deletion, or insertion
        if D[i, j] == pitch_dist + D[i-1, j-1]:
            i -= 1
            j -= 1
        elif D[i, j] == pitch_dist + D[i-1, j]:
            i -= 1
        else:
            j -= 1

    alignment_path.reverse()  # Start-to-end order


    # Get the reference time aligned to the last user note
    last_user_idx, last_ref_idx = alignment_path[-1]
    current_ref_time = ref_times[last_ref_idx]

    return current_ref_time, predicted_speed


