import aubio
import numpy as np
import pyaudio

# Constants for the audio stream
BUFFER_SIZE = 1024  # Number of samples per frame
SAMPLE_RATE = 44100  # Sampling rate in Hz
FORMAT = pyaudio.paFloat32
CHANNELS = 1

def pitch_detection():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open an audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=BUFFER_SIZE)

    # Initialize aubio pitch detection
    pitch_detector = aubio.pitch("default", BUFFER_SIZE * 4, BUFFER_SIZE, SAMPLE_RATE)
    pitch_detector.set_unit("Hz")  # Set pitch unit to Hz
    pitch_detector.set_silence(-40)  # Adjust silence threshold in dB

    print("Listening for pitch (Ctrl+C to stop)...")

    try:
        while True:
            # Read audio data
            audio_data = stream.read(BUFFER_SIZE, exception_on_overflow=False)
            # Convert to aubio-readable format
            samples = np.frombuffer(audio_data, dtype=np.float32)
            # Detect pitch
            pitch = pitch_detector(samples)[0]
            confidence = pitch_detector.get_confidence()

            # Print pitch and confidence
            if confidence > 0.8:  # Adjust confidence threshold if needed
                print(f"Pitch: {pitch:.2f} Hz, Confidence: {confidence:.2f}")
    except KeyboardInterrupt:
        print("\nStopping pitch detection.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    pitch_detection()
