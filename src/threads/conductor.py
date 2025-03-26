import time
import numpy as np
from score_tracking.online_tracking import dtw_pitch_alignment_with_speed, OnlineTracker
 
class Conductor:
    '''
    Tracks the soloist's history and sends signal to accompanist to adjust tempo
    '''

    def __init__(self, solo_events, accomp_player, solo_tracker, adjuster = 'oltw'):
        self.solo_events = solo_events
        self.accomp_player = accomp_player
        self.solo_tracker = solo_tracker
        self.current_solo_index = 0
        self.solo_pitch_history = []
        self.prev_index = 0
        self.prev_pitch = 0


        if adjuster == 'oltw':
            self.adjuster = OnlineTracker(self.solo_events)
        elif adjuster == 'hmm':
            raise NotImplementedError('HMM not implemented yet')
        

    def start(self, barrier, default_sec_per_beat):
        """
        Continuously compares the soloist's performance to the reference track
        and adjusts the accompaniment tempo.
        """
        subdivision = 8

        time_ticker = 1
        soloist_progression = 0
        previous_timing = 0
        
        barrier.wait()
        start_time = time.time()
        while self.accomp_player.playing:
            # prevents excessive tracking
            # this ensures that the pitches detected are mostly accurate and normalized, hopefully eliminating outliers
            if time.time() - start_time < default_sec_per_beat * time_ticker / subdivision:  
                continue

            time_ticker += 1
            latest_pitch = self.solo_tracker.get_latest_pitch()
            accompanist_progression = self.accomp_player.retrieve_progression()
            
            if latest_pitch is None or latest_pitch == 0.0 or latest_pitch == self.prev_pitch:
                continue


            soloist_progression, soloist_index, timing_ratios = self.adjuster.step(np.array([time.time() - start_time, latest_pitch]))
            # print(timing_ratios)
            if timing_ratios is not None:
                print('hi')
                # if not self.solo_pitch_history:
                #     self.solo_pitch_history.append((round(accompanist_progression, 3), latest_pitch))
                # else:
                predicted_past_features = timing_ratios * (accompanist_progression - previous_timing)
                if predicted_past_features is not None:
                    print(predicted_past_features, soloist_index)
                    # print(f'predicted_past_features: {predicted_past_features}')
                # print(time.time() - temp_start)

            previous_timing = accompanist_progression
            self.prev_pitch = latest_pitch

            # print(f'accompanist progression: {accompanist_progression}, soloist progression: {soloist_progression}, soloist index: {soloist_index}')
            # print(f'note played: {self.solo_events[soloist_index]}\n')
            # TODO - Implement tempo adjustment based on final model




            
