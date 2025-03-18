import numpy as np
import librosa
from numpy.typing import NDArray
from .oltw import Euclidean, oltw_loop

class OnlineTracker:
    def __init__(self, 
                 reference_features = None, 
                 window_size = 16, 
                 step_size = 5, 
                 local_cost_fun = Euclidean, 
                 start_window_size = 12):
        self.reference_features = reference_features
        self.last_index = reference_features.shape[0]
        self.window_size = window_size
        self.step_size = step_size
        self.local_cost_fun = local_cost_fun
        self.start_window_size = start_window_size

        self.input_features: NDArray = None
        self.global_cost_matrix = NDArray[np.float32] = (
            np.ones((reference_features.shape[0] + 1, 2)) * np.inf
        )
        self.input_index = 0
        self.update_window_index = False

        self.current_position = 0


    def get_window(self):
        w_size = self.window_size
        if self.current_position < self.start_window_size:
            w_size = self.start_window_size
        window_start = max(self.current_position - w_size, 0)
        window_end = min(self.current_position + w_size, self.last_index)
        return window_start, window_end
    
    def reset(self):
        pass
    
    def step(self, input_features):
        '''
        input_features: np.array of shape (1, 2), where the first element is the time and the second element is the pitch
        '''

        self.input_features = self.input_features + input_features
        self.N_input = self.input_features.shape[0]

        if self.update_window_index:
            self.window_start, self.window_end = self.get_window()
            self.update_window_index = False

        if self.current_position >= self.N_input:
            return None
        

        if self.current_position == 0:
            self.global_cost_matrix[0, 0] = 0
            self.global_cost_matrix[1:self.window_end + 1, 0] = np.cumsum(
                self.local_cost_fun(self.input_features[0], self.reference_features, axis=1)
            )
            return
        
        local_costs = self.local_cost_fun(self.input_features, self.reference_features[self.window_start:self.window_end + 1], axis=1)
        min_costs, min_cost_index = -float('inf'), 0

        cur_checker_index = self.window_start
        while cur_checker_index < self.window_end:
            

            cost1 = self.global_cost_matrix[cur_checker_index, 0] + local_costs[cur_checker_index - self.window_start]
            cost2 = self.global_cost_matrix[cur_checker_index + 1, 0] + local_costs[cur_checker_index - self.window_start + 1]
            cost3 = self.global_cost_matrix[cur_checker_index, 1] + local_costs[cur_checker_index - self.window_start + 1]
            temp_min = min(cost1, cost2, cost3)
            self.global_cost_matrix[cur_checker_index + 1, 1] = temp_min

            norm_cost = temp_min / (cur_checker_index - self.window_start + 1)

            if norm_cost < min_costs:
                min_costs = norm_cost
                min_cost_index = cur_checker_index

            cur_checker_index += 1
        
        self.global_cost_matrix[:, 0] = self.global_cost_matrix[:, 1]
        self.global_cost_matrix[:, 1] = np.inf

        self.current_position = self.current_position = min(
            max(self.current_position, min_cost_index),
            self.current_position + self.step_size,
        )

        return self.reference_features[self.current_position, 1]

        # for i in range(self.window_start, self.window_end):
        #     if i < self.current_position:
        #         continue
        #     if i == self.window_start:
        #         self.global_cost_matrix[0, 1] = self.global_cost_matrix[0, 0] + self.local_cost_fun(
        #             self.input_features[i], self.reference_features[0]
        #         )
        #     else:
        #         self.global_cost_matrix[0, 1] = self.global_cost_matrix[0, 0] + self.local_cost_fun(
        #             self.input_features[i], self.reference_features[0]
        #         )
        #         self.global_cost_matrix[1:, 1] = np.minimum(
        #             self.global_cost_matrix[1:, 0],
        #             np.minimum(
        #                 self.global_cost_matrix[:-1, 0],
        #                 self.global_cost_matrix[:-1, 1],
        #             ),
        #         ) + self.local_cost_fun(self.input_features[i], self.reference_features)
        #     self.global_cost_matrix[:, 0] = self.global_cost_matrix[:, 1]
        # self.current_position += self.step_size
        # if self.current_position >= self.N_input:
        #     return None
        # return self.global_cost_matrix[-1, 0]
    

        

        
        
        




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
            pitch_cost = pitch_distance(user_pitches[i-1], ref_pitches[j-1])
            # pitch_cost = abs(user_pitches[i-1] - ref_pitches[j-1])

            if pitch_cost > pitch_threshold:
                D[i, j] = D[i-1, j]
            else:
                D[i, j] = pitch_cost + min(
                    D[i-1, j],    # Deletion (user note skipped)
                    D[i, j-1],    # Insertion (reference note skipped)
                    D[i-1, j-1]   # Match or substitute
                )
    
    # for i in range(1, I+1):
    #     for j in range(1, J+1):
    #         D[i, j] = D[i, j] / (max(i, j))

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
    predicted_speed, wp, matrix = compute_speed_factor(user_attributes, ref_attributes)


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

