import numpy as np

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



def raw(data, current_index, window_size):
    next_window = data[:, current_index - window_size:current_index].astype(np.float32)
    next_window = next_window - next_window[:, 0:1]

    future_ref = np.array([data[1, current_index]], dtype=np.float32)
    next_window = np.concatenate([next_window.flatten(), future_ref], axis=0)
    # Flatten for model compatibility
    return next_window
    
def difference(data, current_index, window_size):
    # Historical data: differences between consecutive time steps
    first_note = data[:, current_index - window_size:current_index - 1].astype(np.float32)
    second_note = data[:, current_index - window_size + 1:current_index].astype(np.float32)
    historical_data = second_note - first_note
    
    # Future reference: next reference timing difference
    future_ref = np.array([data[1, current_index] - data[1, current_index - 1]], dtype=np.float32)
    
    # Flatten historical data and append future ref
    return np.concatenate([historical_data.flatten(), future_ref])

def difference_first_scaled(data, current_index, window_size):
    first_note = data[:, current_index - window_size:current_index - 1].astype(np.float32)
    second_note = data[:, current_index - window_size + 1:current_index].astype(np.float32)
    historical_data = second_note - first_note
    scale = historical_data[0, 0] / (historical_data[1, 0] + 1e-8)
    historical_data[1] = historical_data[1] * scale
    
    future_ref = np.array([data[1, current_index] - data[1, current_index - 1]], dtype=np.float32) * scale
    print(np.concatenate([historical_data.flatten(), future_ref]))
    return np.concatenate([historical_data.flatten(), future_ref])

def ratio(data, current_index, window_size):
    first_note = data[:, current_index - window_size:current_index - 1].astype(np.float32)
    second_note = data[:, current_index - window_size + 1:current_index].astype(np.float32)
    historical_data = second_note - first_note
    historical_data[0] = historical_data[0] / (historical_data[1] + 1e-8)

    future_ref = np.array([data[1, current_index] - data[1, current_index - 1]], dtype=np.float32)
    return np.concatenate([historical_data.flatten(), future_ref])

def normalized(data, current_index, window_size):
    scale = 1 / 12 / 0.26086426
    first_note = data[:, current_index - window_size:current_index - 1].astype(np.float32)
    second_note = data[:, current_index - window_size + 1:current_index].astype(np.float32)
    historical_data = second_note - first_note
    historical_data[0] = historical_data[0] / (historical_data[1] + 1e-8)
    historical_data[1] = historical_data[1] * scale

    future_ref = np.array([data[1, current_index] - data[1, current_index - 1]], dtype=np.float32) * scale
    return np.concatenate([historical_data.flatten(), future_ref])

def memory_enhanced(data, current_index, window_size):
    scale = 1 / 12 / 0.26086426
    first_note = data[:, current_index - window_size:current_index - 1].astype(np.float32)
    second_note = data[:, current_index - window_size + 1:current_index].astype(np.float32)
    historical_data = second_note - first_note
    historical_data[0] = historical_data[0] / (historical_data[1] + 1e-8)
    historical_data[1] = historical_data[1] * scale

    future_ref = np.array([data[1, current_index] - data[1, current_index - 1]], dtype=np.float32) * scale
    # print(np.array(current_index - window_size))
    return np.concatenate([np.array([current_index - window_size]), historical_data.flatten(), future_ref])

def row_with_ratio(data, current_index, window_size):
    # Historical data: differences between consecutive time steps
    first_note = data[:, current_index - window_size:current_index - 1].astype(np.float32)
    second_note = data[:, current_index - window_size + 1:current_index].astype(np.float32)
    historical_data = second_note - first_note
    # Calculate ratios for row 0
    historical_data[0] = historical_data[0] / (historical_data[1] + 1e-8)
    
    # Future reference: next reference timing difference
    future_ref = np.array([data[1, current_index] - data[1, current_index - 1]], dtype=np.float32)
    
    # Flatten historical data and append future ref
    return np.concatenate([historical_data.flatten(), future_ref])
    
def row_with_next(data, current_index, window_size):
    next_window = data[:, current_index - window_size:current_index].astype(np.float32)
    next_window = next_window - next_window[:, 0:1] 
    # Next prediction note is the future reference
    future_ref = np.array([data[1, current_index]], dtype=np.float32)
    return np.concatenate([next_window.flatten(), future_ref])
    
def forecast(data, current_index, window_size, forecast_window):
    first_note_real = data[0:1, current_index - window_size:current_index - 1].astype(np.float32)
    second_note_real = data[:, current_index - window_size + 1:current_index].astype(np.float32)
    next_window_real = second_note_real - first_note_real

    first_note_ref = data[1:2, current_index - window_size:current_index - 1 + forecast_window].astype(np.float32)
    second_note_ref = data[1:2, current_index - window_size + 1:current_index + forecast_window].astype(np.float32)
    next_window_ref = second_note_ref - first_note_ref

    obs = create_augmented_sequence_with_flags(next_window_real[0], next_window_ref[0], forecast_window).flatten()
    return obs

