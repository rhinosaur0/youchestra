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
        self.prev_onset = None
        self.onset = False
        self.confidences = []

    def start_listening(self):
        self.running = True
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                channels=1, rate=44100, input=True,
                input_device_index=0, frames_per_buffer=512)

        while True:
            data = stream.read(self.hop_size, exception_on_overflow=False)
            samples = np.fromstring(data,dtype=aubio.float_type)        
            pitch = (self.pitch_detector(samples)[0])

            if pitch > 0:
                # self.pitches.put(pitch)
                confidence = self.pitch_detector.get_confidence()
                self.confidences += [confidence]

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

        print(temp)
        return mode(temp) if temp else None
    


    
    
    



# Usage example:
# tracker = SoloTracker()
# try:
#     print("Starting SoloTracker...")
#     tracker.start_listening()
#     while True:
#         pitch = tracker.get_latest_pitch()
#         # if pitch:
#         #     print(f"Detected pitch: {pitch} MIDI")
# except KeyboardInterrupt:
#     print("Stopping SoloTracker...")
#     tracker.stop_listening()
