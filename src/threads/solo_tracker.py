import aubio
import pyaudio
import queue
import numpy as np
from statistics import mode
import time


class SoloTracker:
    '''
    Audio callback is crucial for both ensuring that the accompanist runs smoothly on its thread while also not having any delays in 
    pitch detection.
    '''
    def __init__(self):
        self.samplerate = 44100
        self.hop_size = 256
        self.window_size = 1024  # Adjust as needed for performance
        self.pitch_detector = aubio.pitch("yin", self.window_size, self.hop_size, self.samplerate)
        self.pitch_detector.set_unit("midi")
        self.pitch_detector.set_tolerance(0.2)
        self.pitches = queue.Queue()
        # PyAudio instance will be created in start_listening
        self.pa = pyaudio.PyAudio()
        self.stream = None

    def audio_callback(self, in_data, frame_count, time_info, status_flags):
        # Convert raw data to numpy array of floats.
        samples = np.frombuffer(in_data, dtype=np.float32)
        pitch = self.pitch_detector(samples)[0]
        confidence = self.pitch_detector.get_confidence()
        energy = np.sqrt(np.mean(samples**2))
        if pitch > 0 and energy > 0.015 and confidence > 0.95:
            self.pitches.put(pitch)
            print(f"Detected pitch: {pitch:.2f} (Confidence: {confidence:.2f})")
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

    def get_latest_pitch(self):
        temp = []
        while not self.pitches.empty():
            raw_note = self.pitches.get_nowait()
            temp.append(round(raw_note))
        return mode(temp) if temp else None


    
    
    

