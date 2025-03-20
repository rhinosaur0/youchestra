
import h5py
import numpy as np
import torch as th


def memory_noise(features, noise_std):
    """
    Add Gaussian noise to the features.
    """    
    return features + np.random.randn(*features.shape) * noise_std


def initialize_memory_file(filename="memory.h5", num_piece_indices = 2300, max_runs = 3, obs_dim = 8):
    # Create a new HDF5 file with datasets for memories and run_ids.
    with h5py.File(filename, "w") as f:
        # Create a dataset to store the memory vectors; initialize with zeros.
        f.create_dataset("memories", (num_piece_indices, max_runs, obs_dim), dtype="float32")
        # Create a dataset to store recency scores (run_ids); initialize with very small values.
        f.create_dataset("run_ids", (num_piece_indices, max_runs), dtype="float32", 
                         data=np.full((num_piece_indices, max_runs), -np.inf, dtype="float32"))
    print(f"Initialized memory file '{filename}' with shape memories {(num_piece_indices, max_runs, obs_dim)}.")


def store_memory(piece_index, memory_vector, filename="memory.h5"):
    """
    Stores a memory vector for a given piece index.
    - piece_index: integer index (0 <= piece_index < num_piece_indices)
    - run_id: a recency indicator (larger = more recent)
    - memory_vector: numpy array of shape (obs_dim,)
    This function updates the stored memory so that only the max_runs most recent vectors remain.
    """

    with h5py.File(filename, "a") as f:
        mem_ds = f["memories"]
        run_ds = f["run_ids"]
        # Get the current run_ids for this piece index
        current_run_ids = run_ds[piece_index, :]  # shape (max_runs,)

        # Higher run_id = more recent; find the slot with the smallest run_id
        if np.all(current_run_ids == -np.inf):
            slot = int(np.where(current_run_ids == -np.inf)[0][0])
            new_run_id = 1
        else:
            slot = int(np.argmin(current_run_ids))
            new_run_id = int(np.max(current_run_ids) + 1)
        
        # Update the chosen slot
        mem_ds[piece_index, slot, :] = pad_for_storage(memory_vector.astype("float32"))
        run_ds[piece_index, slot] = new_run_id



def retrieve_memory(piece_index, filename="memory.h5"):
    """
    Retrieves stored memory vectors for a given piece index or a list/array of indices.
    Returns a dictionary mapping each piece index to a list (or array) of memory vectors,
    maintaining the original order of input indices.
    """
    if isinstance(piece_index, th.Tensor):
        piece_index = piece_index.cpu().numpy()
    with h5py.File(filename, "r") as f:
        mem_ds = f["memories"]   # shape: (num_piece_indices, max_runs, obs_dim)
        run_ds = f["run_ids"]    # shape: (num_piece_indices, max_runs)
        
        # Convert input to numpy array and store original order
        if isinstance(piece_index, int):
            original_indices = np.array([piece_index], dtype='i')
            sorted_indices = original_indices
        else:
            original_indices = np.array(piece_index, dtype='i')
            # Get sorting permutation but keep original indices
            sort_perm = np.argsort(original_indices)
            sorted_indices = original_indices[sort_perm]
        
        # Retrieve rows for the sorted indices
        run_ids = run_ds[sorted_indices, :]         # shape: (len(sorted_indices), max_runs)
        memory_vectors = mem_ds[sorted_indices, :]   # shape: (len(sorted_indices), max_runs, obs_dim)
    
    # Build a dictionary mapping each piece index to its sorted memory vectors
    temp_result = {}
    for idx, i in zip(sorted_indices, range(len(sorted_indices))):
        current_run_ids = run_ids[i]       # shape: (max_runs,)
        current_memories = memory_vectors[i]  # shape: (max_runs, obs_dim)
        valid_indices = np.where(current_run_ids != -np.inf)[0]
        
        if len(valid_indices) == 0:
            temp_result[int(idx)] = pad(np.zeros((0, 1)))
        else:
            valid_run_ids = current_run_ids[valid_indices]
            valid_memories = current_memories[valid_indices]
            sorted_order = np.argsort(-valid_run_ids)
            sorted_memories = valid_memories[sorted_order][:, :7]
            temp_result[int(idx)] = pad(sorted_memories)

    result = np.array([temp_result[int(idx)] for idx in original_indices])
    return result
    
def pad(feature):
    '''
    pads feature with 0s to make into [3, 7]
    '''
    return np.pad(feature, ((0, 3 - feature.shape[0]), (0, 7 - feature.shape[1])))

def pad_for_storage(feature):
    '''
    pads feature with 0s to make into [1, 8]
    '''

    new = np.pad(feature, (0, 8 - feature.shape[0]))
    return new



# Example usage:
if __name__ == "__main__":
    pass
    # initialize_memory_file("memory.h5")
    # for i in range(200):
    #     piece_index = i
    #     for j in range(3):
    #         run_id = j + 1
    #         obs_vector = np.random.random(8).astype("float32")
    #         store_memory(piece_index, run_id, obs_vector, "memory.h5")
    # memories = retrieve_memory([5, 2, 8, 1], "memory.h5")
    # print(memories)

