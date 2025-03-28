from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

def compute_energy_scipy(wav_file, frame_size=1024, hop_size=512):
    sample_rate, audio = wavfile.read(wav_file)
    
    # If the audio is stereo (or multi-channel), average the channels to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # If the audio is in integer format, normalize it to [-1, 1]
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    
    energies = []
    for start in range(0, (len(audio) - frame_size)// 100, hop_size):
        frame = audio[start:start + frame_size]
        rms_energy = np.sqrt(np.mean(frame ** 2))
        energies.append(rms_energy)
    
    return np.array(energies), sample_rate

# Example usage:
wav_file = '1729.wav'  # Replace with your file path
energies, sample_rate = compute_energy_scipy(wav_file)

# Calculate time values in seconds for each frame
hop_size = 512  # Make sure this matches the hop_size used in compute_energy_scipy
times = np.arange(len(energies)) * hop_size / sample_rate

# Plot the energy envelope
plt.figure(figsize=(10, 4))
plt.plot(times, energies, label='RMS Energy')
plt.xlabel('Time (seconds)')
plt.ylabel('Energy')
plt.title('Energy Envelope (SciPy)')
plt.legend()
plt.show()
