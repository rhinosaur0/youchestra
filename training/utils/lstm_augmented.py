
def create_augmented_sequence_with_flags(real_diffs, ref_diffs, forecast_horizon):
    """
    real_diffs: numpy array of shape (T,)
    ref_diffs: numpy array of shape (T,)
    forecast_horizon: number of future reference differences to attach
    
    Returns:
      augmented_sequence: numpy array of shape (T - forecast_horizon, 2*(1+forecast_horizon))
      where each element now contains a tuple of (value, flag).
      For the current time stamp, flag=0; for future time stamps, flag=1.
    """
    import numpy as np
    T = len(ref_diffs)
    augmented = []
    for t in range(T - forecast_horizon):
        current_real = [real_diffs[t], 0]
        current_ref = [ref_diffs[t], 0]
        
        future_refs = []
        for i in range(1, forecast_horizon+1):
            future_refs.extend([ref_diffs[t+i], 1])
        
        features = np.array(current_real + current_ref + future_refs)
        augmented.append(features)
    return np.array(augmented)
