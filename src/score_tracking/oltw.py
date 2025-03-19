import numpy as np

def Euclidean(x, y):
    '''
    X is the pitch.
    Y is the reference pitches
    '''
    result = np.array([])
    for _, notes, _ in y:
        if len(notes) >= 1:
            result = np.append(result, np.min([pitch_distance(x, note) for note in notes])) # in the case that there are multiple notes,
                                                                                            # use the smallest distance
        else:
            result = np.append(result, pitch_distance(x, notes[0]))

    return result

def pitch_distance(ref_pitch, perf_pitch):
    semitone_diff = abs(ref_pitch - perf_pitch) % 12
    if semitone_diff == 0:
        return 0  # Exact match
    elif semitone_diff <= 2:
        return 0.5  # Small penalty for nearby notes
    else:
        return 3.0  # Large penalty for outliers (hallucinations)





