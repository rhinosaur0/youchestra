import matplotlib.pyplot as plt
import numpy as np
import aubio
import pyaudio

# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1, rate=44100, input=True,
                input_device_index=0, frames_per_buffer=1024)

# Aubio setup
samplerate = 44100
win_s = 1024  # Larger window size for lower frequency detection
hop_s = 256  # Larger hop size to analyze less frequently

# Onset detector
onset_o = aubio.onset("default", win_s, hop_s, samplerate)
pitch_o = aubio.pitch("default", win_s, hop_s, samplerate)
onset_o.set_threshold(1.0)  # Adjust threshold for reduced sensitivity
pitch_o.set_tolerance(0.4)
pitch_o.set_unit("midi")
# Storage for PCM data and energy
pcm_data = []  # To store audio samples
onset_times = []  # To store detected onset times
time_stamps = []  # To track time for each frame
energy_values = []  # To store energy levels

# Variables for energy comparison
prev_energy = 0
energy_threshold = 0.005  # Larger minimum increase in energy to consider an onset
min_time_gap = 0.1  # Minimum time between onsets (100ms)
last_onset_time = None
energy_cum = [0] * 5

prev_pitch = 0

# Process audio and detect onsets
try:
    frame_count = 0
    while frame_count < 1000:  # Collect 100 frames (~2 seconds of audio)
        data = stream.read(hop_s, exception_on_overflow=False)
        samples = np.fromstring(data, dtype=aubio.float_type)
        pitch = pitch_o(samples)[0]

        # Append PCM data
        pcm_data.extend(samples)

        # Calculate RMS energy
        current_energy = np.sqrt(np.mean(samples**2))
        energy_values.append(current_energy)

        onset = onset_o(samples)
        current_time = frame_count * hop_s / samplerate
        if current_energy > energy_cum[-1] + energy_threshold and pitch > 0:
            # Suppress frequent onsets
            if last_onset_time is None or ((current_time - last_onset_time) > min_time_gap and abs(pitch - prev_pitch) >= 1):
                last_onset_time = current_time
                onset_times.append(current_time)  # Time in seconds
                prev_pitch = pitch
                print(pitch)
                print(f"Onset Detected at {current_time:.2f}s with Energy {current_energy:.4f}")

        # Update previous energy
        energy_cum.append(current_energy)

        # Track time
        time_stamps.append(current_time)

        frame_count += 1
except KeyboardInterrupt:
    pass
finally:
    # Clean up PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

# Convert PCM data to NumPy array
pcm_data = np.array(pcm_data)

# Plot the waveform and energy
plt.figure(figsize=(12, 6))
time_axis = np.linspace(0, len(pcm_data) / samplerate, num=len(pcm_data))
plt.plot(time_axis, pcm_data, label="Waveform", alpha=0.7)
plt.plot(time_stamps, energy_values, label="Energy", color="orange")

# Overlay onsets
for onset_time in onset_times:
    plt.axvline(x=onset_time, color='r', linestyle='--', label='Onset' if onset_time == onset_times[0] else "")

plt.title("Waveform with Detected Onsets (Reduced Frequency)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
