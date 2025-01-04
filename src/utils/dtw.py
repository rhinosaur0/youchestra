import numpy as np
import math


def dtw_pitch_alignment_with_speed(pitch_history, pitch_reference, accompaniment_progress, pitch_threshold=2.0):
    """
    Align pitch_history with pitch_reference using DTW on pitch and predict the speed.
    
    :param pitch_history:  list of (time, pitch_detected)
    :param pitch_reference: list of (time, pitch_expected)
    :param accompaniment_progress: float, the accompaniment progress in seconds
    :param pitch_threshold: float, threshold for average pitch distance confidence
    :return: 
        current_ref_time: float, the reference time aligned to the user's last note
        predicted_speed: float, the cumulative average speed ratio of user vs reference
        confidence: float, a confidence score based on pitch alignment quality
    """
    import numpy as np

    # Preprocess pitch data
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

    # Calculate speed from alignment path
    # print(D)
    # print(alignment_path)
    total_time_ratio = 0.0
    total_weight = 0.0
    last_user_time, last_ref_time = None, None
    for (user_idx, ref_idx) in alignment_path:
        if last_user_time is not None and last_ref_time is not None:
            # Calculate time differences
            user_time_diff = user_times[user_idx] - last_user_time
            ref_time_diff = ref_times[ref_idx] - last_ref_time

            if ref_time_diff > 0:
                local_speed = user_time_diff / ref_time_diff
                weight = 1.0 / (1 + abs(user_pitches[user_idx] - ref_pitches[ref_idx]))  # Weight inversely proportional to pitch distance
                total_time_ratio += local_speed * weight
                total_weight += weight

        # Update last times
        last_user_time = user_times[user_idx]
        last_ref_time = ref_times[ref_idx]

    # Calculate final predicted speed
    predicted_speed = total_time_ratio / total_weight if total_weight > 0 else 1.0
    # print(predicted_speed)

    # Calculate confidence based on average pitch distance
    avg_pitch_dist = np.mean([abs(user_pitches[i] - ref_pitches[j]) for (i, j) in alignment_path])
    confidence = 1.0 if avg_pitch_dist < pitch_threshold else 0.0

    # Get the reference time aligned to the last user note
    last_user_idx, last_ref_idx = alignment_path[-1]
    current_ref_time = ref_times[last_ref_idx]

    return current_ref_time, predicted_speed


