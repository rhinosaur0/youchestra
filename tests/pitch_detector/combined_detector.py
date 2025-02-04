import aubio
from aubio import pitch
import queue
import music21
import pyaudio
import time
import numpy as np
# Open stream.
# PyAudio object.
p = pyaudio.PyAudio()
q = queue.Queue()  
current_pitch = music21.pitch.Pitch()

samplerate = 44100
win_s = 1024
hop_s = 128
tolerance = 0.2

stream = p.open(format=pyaudio.paFloat32,
                channels=1, rate=44100, input=True,
                input_device_index=0, frames_per_buffer=hop_s)

pitch_o = pitch("yin",win_s,hop_s,samplerate)
#pitch_o.set_unit("")
pitch_o.set_tolerance(tolerance)



# total number of frames read
total_frames = 0
def get_current_note():
    pitches = []
    confidences = []
    current_pitch = music21.pitch.Pitch()

    while True:
        data = stream.read(hop_s, exception_on_overflow=False)
        samples = np.fromstring(data,dtype=aubio.float_type)        
        pitch = (pitch_o(samples)[0])
        #pitch = int(round(pitch))
        confidence = pitch_o.get_confidence()
        energy = np.sqrt(np.mean(samples**2))

        # if confidence < 0.1: pitch = 0x
        pitches += [pitch]
        confidences += [confidence]
        current='Nan'
        if pitch > 0 and confidence > 0.8 and energy > 0.015:
            midi_number = 69 + 12 * np.log2(pitch / 440.0)
            current_pitch.frequency = float(pitch)
            current=current_pitch.nameWithOctave
            print(pitch,'----',current,'----',current_pitch.microtone.cents, '----',midi_number, '----', confidence)
        q.put({'Note': current, 'Cents': current_pitch.microtone.cents,'hz':pitch})
        previous_energy = energy
        time.sleep(0.001)
        

if __name__ == '__main__':
    get_current_note()