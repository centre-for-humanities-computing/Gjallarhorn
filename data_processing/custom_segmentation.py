from dataclasses import dataclass
from typing import Union

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class SpeakerSegment:
    start: int = 0
    end: Union[int, None] = None


@dataclass
class SplitStuff4Tw:
    threshold_value: float
    split_index: int


class CustomSegmentationStrategy:

    def __init__(self, analyze_only_seconds: int = 30):
        self.analyze_only_seconds = analyze_only_seconds
        self.min_seconds_preferred = 6
        self.max_seconds_preferred = 15
        self.sampling_rate = 16000
        self.step_size = 200
        self.step_size_seconds = self.step_size / 16000

        self.number_steps = int(self.analyze_only_seconds / self.step_size_seconds)

        self.median_probs = None
        self.trig_sum = None
        self.silence_seconds = 0.3
        self.silence_window_nr_steps = int(self.silence_seconds / self.step_size_seconds)
        self.trigger_window_seconds = 4.0
        self.trigger_window_nr_steps = int(self.trigger_window_seconds / self.step_size_seconds)

    def is_silence(self, buffer_window):
        if np.mean(buffer_window) < self.trig_sum:
            return True

        return False

    def is_above_threshold(self, buffer_window):
        if np.mean(buffer_window) > self.trig_sum:
            return True

        return False

    def convert_steps_to_samples(self, steps):
        # 1 step is 200 samples or self.step_size
        return steps * self.step_size

    def create_better_split_long_length(self, buffer):
        mid_of_clip = int(len(buffer) / 2)
        # 2 seconds each side
        two_seconds = 2 * 16000 / self.step_size

        thresholds = []
        for step_range in range(int(mid_of_clip - two_seconds), int(mid_of_clip + two_seconds),
                                self.silence_window_nr_steps):
            threshold_value = np.mean(buffer[step_range + self.silence_window_nr_steps])

            thresholds.append(SplitStuff4Tw(split_index=int(step_range + self.silence_window_nr_steps / 2),
                                            threshold_value=threshold_value))

        best_split = sorted(thresholds, key=lambda x: x.threshold_value, reverse=False)[0].split_index
        return best_split

    def create_better_split_short_length(self):
        pass

    def segment(self, speaker_vads: np.ndarray):
        self.median = np.median(speaker_vads)
        self.trig_sum = 0.89 * self.median + 0.08

        final_segments = []
        is_speech = False
        current_buffer = []
        temp_speaker_values = None
        for i in range(len(speaker_vads)):
            current_activation = speaker_vads[i]
            current_buffer.append(current_activation)

            if not len(current_buffer) >= self.trigger_window_nr_steps:
                continue

            if not is_speech and self.is_above_threshold(current_buffer):
                is_speech = True
                temp_speaker_values = SpeakerSegment(start=self.convert_steps_to_samples(i - len(current_buffer) + 1))
            elif is_speech:
                # If this but we are not above threshold, check if we are in silence for last steps
                if self.is_silence(buffer_window=current_buffer[:-self.silence_window_nr_steps]):
                    if len(current_buffer) > self.sampling_rate * self.max_seconds_preferred / self.step_size:
                        # find_better split
                        # Todo: Do this recursively
                        split_index = self.create_better_split_long_length(buffer=current_buffer)
                        temp_speaker_values.end = self.convert_steps_to_samples(
                            i - (len(current_buffer) - split_index) - 1)
                        final_segments.append(temp_speaker_values)
                        temp_speaker_values = SpeakerSegment(
                            start=self.convert_steps_to_samples(i - (len(current_buffer) - split_index) + 1),
                            end=self.convert_steps_to_samples(i))
                        final_segments.append(temp_speaker_values)

                        temp_speaker_values = None
                        is_speech = False
                        current_buffer = []
                    elif len(current_buffer) < self.sampling_rate * self.min_seconds_preferred / self.step_size:
                        pass #Since we want at least x seconds, we continue here
                    else:
                        temp_speaker_values.end = self.convert_steps_to_samples(i)
                        final_segments.append(temp_speaker_values)
                        temp_speaker_values = None
                        is_speech = False
                        current_buffer = []
            else:
                # If not above threshold, then keep window constant
                current_buffer.pop(0)

        return final_segments

    def plot_VAD(self, array_yo):
        x = [self.step_size_seconds * i for i in range(self.number_steps)]
        plt.plot(x, array_yo[:self.number_steps])
        plt.show()
