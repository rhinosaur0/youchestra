import aubio
import pyaudio
import queue
import numpy as np
from statistics import mode
import time
import matplotlib.pyplot as plt


class SoloTracker:
    '''
    Pyaudio's audio callback is crucial for both ensuring that the accompanist runs smoothly on its thread 
    while also not having any delays in pitch detection.
    '''
    def __init__(self, debug=True):
        self.samplerate = 44100
        self.hop_size = 256
        self.window_size = 1024  # Adjust as needed for performance
        self.debug = debug
        self.plot = False
        self.onset_threshold = 5000
        
        # Debug storage
        if self.debug:
            self.velocities = []
            self.energies = []
            self.timestamps = []
            self.onset_times = []
            self.start_time = time.time()
        
        # Pitch detection setup
        self.pitch_detector = aubio.pitch("yin", self.window_size, self.hop_size, self.samplerate)
        self.pitch_detector.set_unit("midi")
        self.pitch_detector.set_tolerance(0.2)
        self.pitches = queue.Queue()
        
        # Onset detection setup
        self.onset_detector = aubio.onset("energy", self.window_size, self.hop_size, self.samplerate)
        self.onset_detector.set_threshold(1.5)
        self.prev_onset_time = time.time()
        self.prev_energy = 0
        self.prev_time = 0
        self.onsets = queue.Queue()
        
        # PyAudio instance will be created in start_listening
        self.pa = pyaudio.PyAudio()
        self.stream = None

    def audio_callback(self, in_data, frame_count, time_info, status_flags):
        # Convert raw data to numpy array of floats
        samples = np.frombuffer(in_data, dtype=np.float32)
        
        # Pitch detection
        pitch = self.pitch_detector(samples)[0]
        confidence = self.pitch_detector.get_confidence()
        energy = np.sqrt(np.mean(samples**2))
        
        if pitch > 0 and energy > 0.015 and confidence > 0.95:
            self.pitches.put(pitch)
        
        # Onset detection
        self.onset_detector(samples)
        current_energy = self.onset_detector.get_descriptor()
        current_time = time.time()
        
        # Calculate energy velocity
        if self.prev_time > 0:
            dt = current_time - self.prev_time
            energy_velocity = (current_energy - self.prev_energy) / dt
            
            # Store debug data if enabled
            if self.debug:
                self.velocities.append(energy_velocity)
                self.energies.append(current_energy)
                self.timestamps.append(current_time - self.start_time)
            
            # Detect onset based on velocity threshold and minimum time between onsets
            if energy_velocity > self.onset_threshold and current_time - self.prev_onset_time >= 0.1:
                self.prev_onset_time = current_time
                self.onsets.put(current_time)
                if self.debug:
                    self.onset_times.append(current_time - self.start_time)
        
        # Store current values for next iteration
        self.prev_energy = current_energy
        self.prev_time = current_time
        
        return (None, pyaudio.paContinue)

    def start_listening(self, barrier):
        # Open the stream in callback mode.
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.samplerate,
            input=True,
            frames_per_buffer=self.hop_size,
            stream_callback=self.audio_callback,
            input_device_index=0
        )
        barrier.wait()  # 
        self.stream.start_stream()
        # Keep the main thread alive while the stream is active.
        while self.stream.is_active():
            time.sleep(0.1)

    def stop_listening(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()

        self.plot = True

    def plot_debug_data(self):
        if not self.debug or not self.velocities:
            return
            
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, self.energies, label='Energy')
        plt.title('Onset Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(self.timestamps, self.velocities, label='Velocity')
        plt.axhline(y=7500, color='r', linestyle='--', label='Threshold')
        for onset in self.onset_times:
            plt.axvline(x=onset, color='g', alpha=0.5)
        plt.title('Energy Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def get_latest_pitch(self):
        temp = []
        while not self.pitches.empty():
            raw_note = self.pitches.get_nowait()
            temp.append(round(raw_note))
        return mode(temp) if temp else None

    def get_latest_onset(self):
        return self.onsets.get_nowait() if not self.onsets.empty() else None






