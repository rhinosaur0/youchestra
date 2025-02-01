import rtmidi

midi_out = rtmidi.MidiOut()
ports = midi_out.get_ports()
print("Available MIDI Ports:")
for i, port in enumerate(ports):
    print(f"{i}: {port}")