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
    if semitone_diff <= 1:
        return 0.5  # Exact match
    else:
        return 4.0  



def pad(features):
    """
    Pad a 2D array to have 7 features while preserving the number of rows.
    
    Args:
        features: List of arrays or 2D numpy array with shape (2, n) where n < 7
        
    Returns:
        Padded array with shape (2, 7)
    """
    features = np.array(features)
    current_features = features.shape[1]
    padding_width = ((0, 0), (0, 7 - current_features))  # ((row_before, row_after), (col_before, col_after))
    return np.pad(features, padding_width, mode="constant")





