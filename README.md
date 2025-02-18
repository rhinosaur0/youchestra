# Youchestra

A reinforcement learning project to create a dynamic music accompanist that adjusts its timing to synchronize with a soloist’s performance. The agent uses a reference MIDI file, historical performance data, and a recurrent PPO (Recurrent Proximal Policy Optimization) model with an LSTM policy to predict an optimal speed adjustment factor.

## Overview

This project is designed to help a virtual accompanist learn to follow a soloist by predicting speed adjustments based on:
- **Reference MIDI Data:** Contains a reference performance (e.g., an accompaniment track).
- **Soloist MIDI Data:** Contains performance timings from the soloist.
- **Augmented Observations:** Observations include note events (pitch, real time, reference time) and can be augmented with bar line markers to indicate measure boundaries.

The agent is trained using a custom Gym environment and the [RecurrentPPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) algorithm from the `sb3_contrib` package.

## Features
- Soloist melody detection and accompanist playback using Mido
  Created Dynamic Time Warping algorithm to accurately track the soloist's location in the performance in reference to the soloist MIDI file.

- **Custom Gym Environment:**  
  Provides a sliding window of observations that include performance times and optional bar line markers.

- **Recurrent PPO Agent:**  
  Utilizes an LSTM policy to capture sequential musical context over time.
