# Youchestra

A reinforcement learning project to create a dynamic music accompanist that adjusts its timing to synchronize with a soloist’s performance. The agent uses a reference MIDI file, historical performance data, and a recurrent PPO (Recurrent Proximal Policy Optimization) model with an LSTM policy to predict an optimal speed adjustment factor.

## Inspiration

I never had the opportunity to perform with an orchestra as a pianist. I want amateur musicians to experience what it's like to perform alongside an orchestra. 

## Overview

This project is designed to create an orchestral accompanist to follow a soloist by predicting speed adjustments based on:
- **Reference MIDI Data:** Contains a reference performance, i.e. a metronomic midi file of the soloist section.
- **Soloist MIDI Data:** Contains live performance timings from the soloist.
- **Augmented Observations:** Observations include note events (pitch, real time, reference time) and can be augmented with bar line markers to indicate measure boundaries.

The agent is trained using a custom Gym environment and the [RecurrentPPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) algorithm from the `sb3_contrib` package.

## Features
- **Soloist melody detection and accompanist playback using Mido**
  Designed Online Time Warping (OLTW) algorithm to accurately track the soloist's location in the performance in reference to the soloist MIDI file.

- **Custom Gym Environment:**  
  Provides a sliding window of observations that include performance times and optional bar line markers. The agent uses the previous window to predict the speed change of the soloist in the next few notes. 

- **Recurrent PPO Agent:**  
  Utilizes an LSTM policy to capture sequential musical context over time.

- **Memory augmentation:**
  A soloist's timing is not deterministic, and every soloist likes to treat music differently. Therefore, the model will remember previous soloist runthroughs, and use memory features to help decision making.

## Current Progress
- Finished and tested complete soloist-detection, accompanist playback, and OLTW algorithm on Mendelssohn Violin Concerto Op. 64.
- Finished data cleaning on Chopin's Ballade No. 1. This is the training data I used for the current RPPO agent, because this ballade has a diverse range of musical expressions that will help the agent better understand a soloist. In reality, the model may need to be finetuned to specific genres. I used the live performance MIDI file from [The MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) and the default, metronomic performance MIDI file from [Online Sequencer](https://onlinesequencer.net/1771756). I extracted the melody lines to train the agent.
- March 10th - Finished creating the environment and the training pipeline.
- March 15th - Designed custom RPPO model architecture to accommodate for time series data (read src/rl/custom_network.py for more details)
- March 19th - Working model (could use fine tuning), ready for integration 
- March 26th - Discovered that onset detection is really accurate as long as the audio source is not the device running the script


## TODO
- Accurate soloist timings extraction to be integrated with model.


References:

Online Time Warping inspiration: 
https://github.com/CPJKU/accompanion
