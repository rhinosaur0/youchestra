import time
from utils.dtw import dtw_pitch_alignment_with_speed
 
class Conductor:
    def __init__(self, solo_events, accomp_player, solo_tracker):
        self.solo_events = solo_events
        self.accomp_player = accomp_player
        self.solo_tracker = solo_tracker
        self.current_solo_index = 0
        self.solo_pitch_history = []
        

    def start(self, barrier, default_sec_per_beat):
        """
        Continuously compares the soloist's performance to the reference track
        and adjusts the accompaniment tempo dynamically.
        """
        subdivision = 8
        start_time = time.time()
        time_ticker = 1
        solo_pitch_reference = [note for note in self.solo_events[:32] if note[0] < 8 * default_sec_per_beat]
        soloist_first_event = 0
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

            if not self.solo_pitch_history or abs(latest_pitch - self.solo_pitch_history[-1][1]) >= 1:
                self.solo_pitch_history.append((round(time.time() - start_time, 2), latest_pitch))
                self.solo_pitch_history = [item for item in self.solo_pitch_history if \
                                          item[0] + default_sec_per_beat * 4 >= accompanist_progression]
            else:
                continue
            
            print(self.solo_pitch_history)
            soloist_progression, predicted_speed = dtw_pitch_alignment_with_speed(self.solo_pitch_history, solo_pitch_reference, accompanist_progression)
            print(f'accompanist progression: {accompanist_progression}, soloist progression: {soloist_progression}')
            
            new_window = []
            i = soloist_first_event - 32
            while i < 0 or self.solo_events[i][0] < accompanist_progression - 8 * default_sec_per_beat:
                i += 1
            while self.solo_events[i][0] < accompanist_progression + 8 * default_sec_per_beat and i - soloist_first_event < 32:
                new_window.append(self.solo_events[i])
                i += 1
            soloist_first_event = i
            solo_pitch_reference = new_window
            # if not solo_pitch_reference:
            #     print(solo_pitch_reference)
        

            # if abs(latest_pitch - ref_pitch[0]) < 1:  # Allow some tolerance
            #     self.current_solo_index += 1
            #     if self.current_solo_index >= len(self.solo_events):
            #         break  # Soloist finished the performance
            #     self.accomp_player.adjust_tempo(1.0)
            # else:
            #     # Adjust tempo if soloist is behind or ahead
            #     if latest_pitch > ref_pitch[0]:
            #         self.accomp_player.adjust_tempo(1.2)  # Speed up
            #     else:
            #         self.accomp_player.adjust_tempo(0.8)  # Slow down

            
