from enum import Enum

import ffmpeg
import numpy as np
import pandas as pd
import torch

from data_processing.custom_segmentation import CustomSegmentationStrategy
from data_processing.voice_activity_detection import VADSilero


class Method(Enum):
    CUSTOM = "CUSTOM"
    SILERO = "SILERO"


class AudioConvert:

    def __init__(self, method: Method = Method.CUSTOM):
        self.custom_segmentation = CustomSegmentationStrategy()
        self.custom_speaker_activity_detection = VADSilero()
        self.method = method

    @staticmethod
    def read_file_to_np(audiofile_path: str):
        out, err = (
            ffmpeg
                .input(audiofile_path)
                .output('pipe:', format="wav", acodec="pcm_s16le", ar=16000, ac=1)
                .run(capture_stdout=True)
        )
        numpy_array = np.frombuffer(out, dtype=np.int16)
        return numpy_array

    def convert_file_to_segments(self, audiofile_path: str):
        audio = self.read_file_to_np(audiofile_path)
        audio_tensor = torch.Tensor(audio)

        if self.method == Method.CUSTOM:
            vad_matrix = self.custom_speaker_activity_detection.get_VAD_matrix(audio_tensor)
            self.custom_segmentation.plot_VAD(vad_matrix)
            segments = self.custom_segmentation.segment(vad_matrix.numpy())
            audio_segments = self.custom_speaker_activity_detection.audio_to_segments_from_stamps(audio, segments)
        elif self.method == Method.SILERO:
            timestamps = self.custom_speaker_activity_detection._get_speech_ts_adaptive(audio_tensor)
            audio_segments = self.custom_speaker_activity_detection.audio_to_segments(audio, timestamps)
        else:
            raise RuntimeError()

        return audio_segments

if __name__ == '__main__':
    method = Method.SILERO
    converter = AudioConvert(method=method)
    audio_files = [
        #"/media/rafje/danspeech/data_mining/unlabeled/podcasts/foelg_pengende/Foelg-pengene--Hvem-sk_5e5eee8c464747fdaab37a30a626df9b_192.mp3",
        #"/media/rafje/danspeech/data_mining/unlabeled/podcasts/24_spørgsmål_til_professoren/Historier_fra_de_varme_lande.mp3",
        #"/media/rafje/danspeech/data_mining/unlabeled/podcasts/danske_statsministre/Bang_Andr_f_rdigproduceret_med_intro_og_outro_online-audio-converter_com_.mp3",
        #"/media/rafje/danspeech/data_mining/unlabeled/podcasts/den_agile_podcast/Podcast#3 - Agile kontra vandfald.mp3",
        #"/media/rafje/danspeech/data_mining/unlabeled/podcasts/supertanker/Supertanker--USA-paa-r_2c271306def14480840af87150e5d636_192.mp3",
        "/home/rafje/Downloads/Foelg-pengene--Apple--_823566a09c664d17aad77862d288473a_192.mp3"
    ]

    audio_lenghts = []
    for audio_file in audio_files:
        lengths = map(lambda x: len(x[2]) / 16000, converter.convert_file_to_segments(audio_file))
        audio_lenghts.append(lengths)

    import matplotlib.pyplot as plt

    all_lengths = []

    lower_seconds = 4
    upper_seconds = 15

    under_seconds = []
    between = []
    over_seconds = []

    for i in range(len(audio_lenghts)):
        current_lengths = list(audio_lenghts[i])
        all_lengths += current_lengths
        df = pd.DataFrame(current_lengths, columns=['one'])
        ax = df.plot.hist(bins=20, alpha=0.5)
        plt.show()

        for audio_length in current_lengths:
            if audio_length < lower_seconds:
                under_seconds.append(audio_length)
            if audio_length > upper_seconds:
                over_seconds.append(audio_length)
            else:
                between.append(audio_length)

    df = pd.DataFrame(all_lengths, columns=['Audio lengths'])
    ax = df.plot.hist(bins=20, alpha=0.5)
    plt.show()

    print(f"Length under: {len(under_seconds)}")
    print(f"Length over: {len(over_seconds)}")
    print(f"Length between: {len(between)}")
    print(f"total length: {len(under_seconds) + len(over_seconds) + len(between)}")

    print(f"Length under seconds: {sum(under_seconds)}")
    print(f"Length over seconds: {sum(over_seconds)}")
    print(f"Length between seconds: {sum(between)}")

    print(f"total length seconds: {sum(under_seconds) + sum(over_seconds) + sum(between)}")
