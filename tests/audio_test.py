import rtmidi
import time

# Initialize MIDI output
midi_out = rtmidi.MidiOut()
ports = midi_out.get_ports()

if not ports:
    print("No MIDI output ports available.")
    exit()

# Open the IAC Driver port
for i, port in enumerate(ports):
    print(f"Port {i}: {port}")
    if "IAC Driver" in port:
        print(f"Using MIDI Port: {port}")
        midi_out.open_port(i)
        break
else:
    print("No suitable MIDI port found.")
    exit()

# Send a Note On and Note Off
try:
    print("Sending Note On...")
    midi_out.send_message([0x90, 60, 100])  # Note On: middle C
    time.sleep(1)
    print("Sending Note Off...")
    midi_out.send_message([0x80, 60, 0])    # Note Off: middle C
finally:
    midi_out.close_port()