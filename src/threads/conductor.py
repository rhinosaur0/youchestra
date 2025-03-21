import time
import numpy as np
from score_tracking.online_tracking import dtw_pitch_alignment_with_speed, OnlineTracker
 
class Conductor:
    def __init__(self, solo_events, accomp_player, solo_tracker):
        self.solo_events = solo_events
        self.accomp_player = accomp_player
        self.solo_tracker = solo_tracker
        self.current_solo_index = 0
        self.solo_pitch_history = []

        self.adjuster = OnlineTracker(self.solo_events)
        

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
            # universal time of the piece progression
            accompanist_progression = self.accomp_player.retrieve_progression()
            # print(latest_pitch)
            
            if latest_pitch is None or latest_pitch == 0.0:
                continue

            # update solo_pitch_history if conditions are met
            # if not self.solo_pitch_history or abs(latest_pitch - self.solo_pitch_history[-1][1]) >= 1:
            #     self.solo_pitch_history.append((round(time.time() - start_time, 2), latest_pitch))
            #     self.solo_pitch_history = [
            #         item for item in self.solo_pitch_history
            #         if item[0] + default_sec_per_beat * 4 >= accompanist_progression
            #     ]
            # else:
            #     continue

            soloist_progression, soloist_index, timing_ratios = self.adjuster.step(np.array([time.time() - start_time, latest_pitch]))
            if timing_ratios is not None:
                if not self.solo_pitch_history:
                    self.solo_pitch_history.append((round(accompanist_progression, 3), latest_pitch))
                else:
                    predicted_past_features = timing_ratios * (accompanist_progression - previous_timing)
                    # print(f'predicted_past_features: {predicted_past_features}')
                # print(time.time() - temp_start)

                previous_timing = accompanist_progression

            print(f'accompanist progression: {accompanist_progression}, soloist progression: {soloist_progression}, soloist index: {soloist_index}\n')

            # TODO - Implement tempo adjustment based on final model




            
