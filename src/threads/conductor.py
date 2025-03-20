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
        subdivision = 4
        start_time = time.time()
        time_ticker = 1
        solo_pitch_reference = [note for note in self.solo_events[:16] if note[0] < 8 * default_sec_per_beat]
        soloist_first_event = 0
        soloist_progression = 0
        
        barrier.wait()
        while self.accomp_player.playing:
            # prevents excessive tracking
            # this ensures that the pitches detected are mostly accurate and normalized, hopefully eliminating outliers
            if time.time() - start_time < default_sec_per_beat * time_ticker / subdivision:  
                continue

            time_ticker += 1
            latest_pitch = self.solo_tracker.get_latest_pitch()
            
            if latest_pitch is None or latest_pitch == 0.0:
                continue
            
            # universal time of the piece progression
            accompanist_progression = self.accomp_player.retrieve_progression()

            if not hasattr(self, 'cached_solo_pitch_start_time'):
                self.cached_solo_pitch_start_time = None  # Initialize cache

            # update solo_pitch_history if conditions are met
            if not self.solo_pitch_history or abs(latest_pitch - self.solo_pitch_history[-1][1]) >= 1:
                self.solo_pitch_history.append((round(time.time() - start_time, 2), latest_pitch))
                self.solo_pitch_history = [
                    item for item in self.solo_pitch_history
                    if item[0] + default_sec_per_beat * 4 >= accompanist_progression
                ]
            else:
                continue

            print(self.solo_pitch_history)

            soloist_progression = self.adjuster.step(np.array([time.time() - start_time, latest_pitch]))

            # print(self.solo_pitch_history)
            # print(solo_pitch_reference)

            # Check if the start of the solo pitch history has changed
            # if not self.cached_solo_pitch_start_time or self.cached_solo_pitch_start_time != self.solo_pitch_history[0][0]:
            #     self.cached_solo_pitch_start_time = self.solo_pitch_history[0][0]

            #     new_window = []
            #     i = soloist_first_event - 16
            #     while i < 0 or self.solo_events[i][0] < soloist_progression - 8 * default_sec_per_beat: i += 1
            #     while self.solo_events[i][0] < soloist_progression + 8 * default_sec_per_beat and i - soloist_first_event < 16:
            #         new_window.append(self.solo_events[i])
            #         i += 1
            #     soloist_first_event = i
            #     solo_pitch_reference = new_window

            # soloist_progression, predicted_speed = dtw_pitch_alignment_with_speed(
            #     self.solo_pitch_history, solo_pitch_reference, accompanist_progression
            # )
            print(f'accompanist progression: {accompanist_progression}, soloist progression: {soloist_progression}\n')

            # TODO - Implement tempo adjustment based on final model




            
