
import mido
import time

# Open the first available MIDI output port
print(mido.get_output_names())

midi_out = mido.open_output()

# Define a chord (C Major: C-E-G)
chord_notes = [60, 64, 67]  # MIDI pitches for C, E, G

# Create MIDI messages for all notes (Velocity 100)
chord_messages = [mido.Message('note_on', note=n, velocity=100) for n in chord_notes]

# 🚀 Send all notes in one batch (faster, minimal delay)
for msg in chord_messages:
    midi_out.send(msg)

time.sleep(1)  # Hold for 1 second

# Send Note Off messages in one batch
for msg in chord_messages:
    midi_out.send(mido.Message('note_off', note=msg.note, velocity=0))