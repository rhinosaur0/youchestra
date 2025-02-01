from sklearn.linear_model import LinearRegression
import numpy as np

def train_linear_regression(data):
    """
    Train a linear regression model for predicting tempo based on relative timing features.

    :param data: list of tuples [(features, target_speed)], where:
                 - features: [relative_interval_1, relative_interval_2, ..., relative_interval_n]
                 - target_speed: the true tempo of the soloist (relative to reference)
    :return: trained linear regression model
    """
    # Extract features (X) and targets (y)
    X = np.array([0] + [d[0] for d in data]).reshape(-1, 1)  # Features: relative timing intervals
    y = np.array([0] + [d[1] for d in data])  # Target speeds

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]

def predict_tempo_with_linear_regression(model, alignment_path, user_times, ref_times, num_recent=5):
    """
    Predict the tempo of the soloist using a trained linear regression model.

    :param model: trained linear regression model
    :param alignment_path: list of (i, j) tuples representing aligned notes
    :param user_times: array of timestamps for soloist notes
    :param ref_times: array of timestamps for reference notes
    :param num_recent: number of recent notes to consider for prediction
    :return: predicted_speed: the relative tempo of the soloist
    """
    # Extract the most recent notes from the alignment path
    if len(alignment_path) < 2:
        return 1.0
    
    relative_intervals = []

    start_i, start_j = alignment_path[0]

    for k in range(1, len(alignment_path)):
        i_curr, j_curr = alignment_path[k]

        # Only compute the ratio when both i and j change
        if i_curr != start_i and j_curr != start_j:
            # Compute time intervals
            user_interval = user_times[i_curr] - user_times[start_i]
            ref_interval = ref_times[j_curr] - ref_times[start_j]

            # Avoid division by zero
            if ref_interval > 0:
                relative_intervals.append(user_interval / ref_interval)
            
            # Update starting point
            start_i, start_j = i_curr, j_curr


    # Pad relative_intervals to ensure it matches num_recent-1 length
    while len(relative_intervals) < num_recent - 1:
        relative_intervals.insert(0, 1.0)  # Assume neutral tempo for missing intervals

    # Predict the tempo using the linear regression model
    features = np.array(relative_intervals).reshape(1, -1)  # Shape as a single sample
    predicted_speed = 1

    return predicted_speed

