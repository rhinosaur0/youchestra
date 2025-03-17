import numpy as np

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
