import torch as th
import h5py
import numpy as np


def memory_noise(features, noise_std):
    """
    Add Gaussian noise to the features.
    """
    return features + th.randn_like(features) * noise_std




def initialize_memory_file(filename="memory.h5", num_piece_indices = 2100, max_runs = 3, obs_dim = 8):
    # Create a new HDF5 file with datasets for memories and run_ids.
    with h5py.File(filename, "w") as f:
        # Create a dataset to store the memory vectors; initialize with zeros.
        f.create_dataset("memories", (num_piece_indices, max_runs, obs_dim), dtype="float32")
        # Create a dataset to store recency scores (run_ids); initialize with very small values.
        f.create_dataset("run_ids", (num_piece_indices, max_runs), dtype="float32", 
                         data=np.full((num_piece_indices, max_runs), -np.inf, dtype="float32"))
    print(f"Initialized memory file '{filename}' with shape memories {(num_piece_indices, max_runs, obs_dim)}.")



def store_memory(piece_index, run_id, memory_vector, filename="memory.h5"):
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
        if np.any(current_run_ids == -np.inf):
            slot = int(np.where(current_run_ids == -np.inf)[0][0])
        else:
            slot = int(np.argmin(current_run_ids))
            if run_id <= current_run_ids[slot]:
                return
        
        # Update the chosen slot
        mem_ds[piece_index, slot, :] = memory_vector.astype("float32")
        run_ds[piece_index, slot] = run_id

    print(f"Stored memory for piece index {piece_index} in slot {slot} with run_id {run_id}.")


def retrieve_memory(piece_index, filename="memory.h5"):
    """
    Retrieves stored memory vectors for a given piece index or a list/array of indices.
    Returns a dictionary mapping each piece index to a list (or array) of memory vectors,
    sorted by recency (most recent first).
    """
    with h5py.File(filename, "r") as f:
        mem_ds = f["memories"]   # shape: (num_piece_indices, max_runs, obs_dim)
        run_ds = f["run_ids"]    # shape: (num_piece_indices, max_runs)
        
        # Ensure piece_index is an array of sorted indices (ascending order)
        if isinstance(piece_index, int):
            indices = np.array([piece_index], dtype='i')
        else:
            indices = np.sort(np.array(piece_index, dtype='i'))
        
        # Retrieve rows for the requested indices
        run_ids = run_ds[indices, :]         # shape: (len(indices), max_runs)
        memory_vectors = mem_ds[indices, :]    # shape: (len(indices), max_runs, obs_dim)
    
    # Build a dictionary mapping each piece index to its sorted memory vectors
    result = {}
    for idx, i in zip(indices, range(len(indices))):
        # For each piece index, filter out empty slots (where run_id is -inf)
        current_run_ids = run_ids[i]       # shape: (max_runs,)
        current_memories = memory_vectors[i]  # shape: (max_runs, obs_dim)
        valid_indices = np.where(current_run_ids != -np.inf)[0]
        
        if len(valid_indices) == 0:
            result[int(idx)] = []
        else:
            valid_run_ids = current_run_ids[valid_indices]
            valid_memories = current_memories[valid_indices]
            # Sort in descending order of run_id (most recent first)
            sorted_order = np.argsort(-valid_run_ids)
            sorted_memories = valid_memories[sorted_order]
            result[int(idx)] = sorted_memories  # You could also return (run_ids, memories) if desired
    
    return result

# Example usage:
# if __name__ == "__main__":
    # initialize_memory_file("memory.h5")
    

    # for i in range(200):
    #     piece_index = i
    #     for j in range(3):
    #         run_id = j + 1
    #         obs_vector = np.random.random(8).astype("float32")
    #         store_memory(piece_index, run_id, obs_vector, "memory.h5")
    
    # memories = retrieve_memory(piece_index, "memory.h5")

