import torch as th

def memory_noise(features, noise_std):
    """
    Add Gaussian noise to the features.
    """
    return features + th.randn_like(features) * noise_std


        
import h5py
import numpy as np

# Configuration parameters
num_piece_indices = 2100  # Number of distinct locations in the piece
max_runs = 3              # Maximum runs to keep per piece index
obs_dim = 8               # Dimension of each observation vector (e.g., window_size+1)

def initialize_memory_file(filename="memory.h5"):
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
        # If there is an empty slot (e.g., still -inf) or the new run_id is more recent than the oldest,
        # find the slot to replace.
        if np.any(current_run_ids == -np.inf):
            slot = int(np.where(current_run_ids == -np.inf)[0][0])
        else:
            # Otherwise, replace the one with the smallest run_id if the new run is more recent.
            slot = int(np.argmin(current_run_ids))
            if run_id <= current_run_ids[slot]:
                # The new memory is not more recent than any stored memory; optionally, skip storage.
                return
        
        # Update the chosen slot
        mem_ds[piece_index, slot, :] = memory_vector.astype("float32")
        run_ds[piece_index, slot] = run_id
        # (Optionally, sort the runs by recency here if needed.)
    print(f"Stored memory for piece index {piece_index} in slot {slot} with run_id {run_id}.")

def retrieve_memory(piece_index, filename="memory.h5"):
    """
    Retrieves stored memory vectors for a given piece index.
    Returns a list of (run_id, memory_vector) sorted by recency (most recent first).
    """
    with h5py.File(filename, "r") as f:
        mem_ds = f["memories"]
        run_ds = f["run_ids"]
        run_ids = run_ds[piece_index, :]  # shape (max_runs,)
        memory_vectors = mem_ds[piece_index, :]  # shape (max_runs, obs_dim)
    
    # Filter out empty slots (-inf run_id)
    valid_indices = np.where(run_ids != -np.inf)[0]
    if len(valid_indices) == 0:
        return []
    
    valid_run_ids = run_ids[valid_indices]
    valid_memories = memory_vectors[valid_indices]
    
    # Sort by run_id descending (most recent first)
    sorted_order = np.argsort(-valid_run_ids)
    sorted_run_ids = valid_run_ids[sorted_order]
    sorted_memories = valid_memories[sorted_order]
    
    return list(zip(sorted_run_ids, sorted_memories))

# Example usage:
if __name__ == "__main__":
    # Initialize the file (run this once)
    initialize_memory_file("memory.h5")
    
    # Store a sample memory
    piece_index = 240
    run_id = 1234567890.0  # e.g., a timestamp or counter
    obs_vector = np.random.random(obs_dim).astype("float32")
    store_memory(piece_index, run_id, obs_vector, "memory.h5")
    
    # Retrieve memories for that piece index
    memories = retrieve_memory(piece_index, "memory.h5")
    print("Retrieved memories for piece index", piece_index)
    for rid, vec in memories:
        print(f"Run ID: {rid}, Vector: {vec}")

