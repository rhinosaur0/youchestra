import matplotlib.pyplot as plt
import numpy as np
import aubio
import pyaudio
import time

# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1, rate=44100, input=True,
                input_device_index=0, frames_per_buffer=1024)

# Aubio setup
samplerate = 44100
win_s = 1024
hop_s = 256

# Create onset detector with both energy and spectral diff for comparison
onset_energy = aubio.onset("energy", win_s, hop_s, samplerate)
onset_specdiff = aubio.onset("specflux", win_s, hop_s, samplerate)
# Set thresholds
onset_energy.set_threshold(1.5)
onset_specdiff.set_threshold(0.1)

prev_onset = time.time()

# Storage for values
energy_values = []
specdiff_values = []
onset_energy_times = []
onset_specdiff_times = []
time_stamps = []
energy_velocities = []
specdiff_velocities = []

# Get the underlying onset detection function values
try:
    frame_count = 0
    prev_energy = 0
    prev_specdiff = 0
    prev_time = 0
    
    while frame_count < 4000:
        data = stream.read(hop_s, exception_on_overflow=False)
        samples = np.fromstring(data, dtype=aubio.float_type)
        
        current_time = frame_count * hop_s / samplerate
        
        # Get both energy and spectral difference values
        onset_energy(samples)
        onset_specdiff(samples)
        
        # Get the actual detection function values
        energy_val = onset_energy.get_descriptor()
        specdiff_val = onset_specdiff.get_descriptor()
        
        # Calculate instantaneous velocity if we have previous values
        if frame_count > 0:
            dt = current_time - prev_time
            energy_velocity = (energy_val - prev_energy) / dt
            specdiff_velocity = (specdiff_val - prev_specdiff) / dt
            
            # Store velocities
            energy_velocities.append(energy_velocity)
            specdiff_velocities.append(specdiff_velocity)
            
            # Detect onsets based on velocity threshold
            if energy_velocity > 7500 and time.time() - prev_onset >= 0.1:
                prev_onset = time.time()
                onset_energy_times.append(current_time)

        else:
            energy_velocities.append(0)
            specdiff_velocities.append(0)
        
        # Store current values for next iteration
        prev_energy = energy_val
        prev_specdiff = specdiff_val
        prev_time = current_time
        
        # Store values
        energy_values.append(energy_val)
        specdiff_values.append(specdiff_val)
        time_stamps.append(current_time)
        
        frame_count += 1
        
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

# Plot the detection functions and their velocities
plt.figure(figsize=(15, 8))

# Plot energy detection and its velocity
plt.subplot(2, 1, 1)
plt.plot(time_stamps, energy_values, 'b-', label='Energy Detection Function')
plt.plot(time_stamps, energy_velocities, 'c-', label='Energy Velocity', alpha=0.7)
plt.axhline(y=7500, color='r', linestyle='--', label='Velocity Threshold')
for onset in onset_energy_times:
    plt.axvline(x=onset, color='g', alpha=0.5)
plt.title('Energy-based Onset Detection')
plt.xlabel('Time (s)')
plt.ylabel('Energy / Velocity')
plt.legend()

# Plot spectral difference detection and its velocity
plt.subplot(2, 1, 2)
plt.plot(time_stamps, specdiff_values, 'b-', label='Spectral Difference Function')
plt.plot(time_stamps, specdiff_velocities, 'm-', label='Spectral Diff Velocity', alpha=0.7)
plt.axhline(y=7500, color='r', linestyle='--', label='Velocity Threshold')
for onset in onset_specdiff_times:
    plt.axvline(x=onset, color='g', alpha=0.5)
plt.title('Spectral Difference Onset Detection')
plt.xlabel('Time (s)')
plt.ylabel('Spectral Difference / Velocity')
plt.legend()

plt.tight_layout()
plt.show()
