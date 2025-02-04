import aubio
import pyaudio
import queue
import numpy as np
from statistics import mode
import time

class SoloTracker:
    def __init__(self):
        self.samplerate = 44100
        self.hop_size = 512
        self.window_size = 1024  # Adjusted buffer size for better performance
        
        self.pitch_detector = aubio.pitch("yin", self.window_size, self.hop_size, self.samplerate)
        self.pitch_detector.set_unit("midi")
        self.pitch_detector.set_tolerance(0.2)

        self.running = False
        self.pitches = queue.Queue()
        self.prev_onset = 0
        self.onset = False
        self.confidences = []

    def start_listening(self, barrier):
        self.running = True
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                channels=1, rate=44100, input=True,
                input_device_index=0, frames_per_buffer=self.hop_size)

        barrier.wait()
        while True:
            data = stream.read(self.hop_size, exception_on_overflow=False)
            samples = np.fromstring(data,dtype=aubio.float_type)        
            pitch = (self.pitch_detector(samples)[0])
            confidence = self.pitch_detector.get_confidence()
            energy = np.sqrt(np.mean(samples**2))

            if pitch > 0 and energy > 0.015 and confidence > 0.8:
                self.pitches.put(pitch)
                confidence = self.pitch_detector.get_confidence()
                self.confidences += [confidence]
                # print(pitch, confidence)
            time.sleep(0.0001)

    def stop_listening(self):
        self.running = False
        if hasattr(self, "stream"):
            self.stream.stop_stream()
            self.stream.close()

    # external
    def get_latest_pitch(self):
        temp = []
        while not self.pitches.empty():
            raw_note = self.pitches.get_nowait()
            temp.append(round(raw_note))
        # print(temp)
        return mode(temp) if temp else None
    


    
    
    

