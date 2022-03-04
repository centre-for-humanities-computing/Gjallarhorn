import ffmpeg
import numpy as np
import soundfile as sf
import torch

from data_processing.custom_segmentation import CustomSegmentationStrategy
from data_processing.voice_activity_detection import VADSilero


class AudioConvert:

    def __init__(self):
        self.custom_segmentation = CustomSegmentationStrategy()
        self.custom_speaker_activity_detection = VADSilero()

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
        vad_matrix = self.custom_speaker_activity_detection.get_VAD_matrix(audio_tensor)
        segments = self.custom_segmentation.segment(vad_matrix.numpy())
        audio_segments = self.custom_speaker_activity_detection.audio_to_segments_from_stamps(audio, segments)
        self.segments_to_files(audio_segments)
        return segments

    def segments_to_files(self, audio_segments):
        for i, segment in enumerate(audio_segments):
            sf.write(f"../test_files/{i}.wav", segment[2], samplerate=16000)


if __name__ == '__main__':
    converter = AudioConvert()
    audio_file = "/media/rafje/danspeech/data_mining/unlabeled/podcasts/foelg_pengende/Foelg-pengene--Hvem-sk_5e5eee8c464747fdaab37a30a626df9b_192.mp3"
    segments = converter.convert_file_to_segments(audio_file)
    print(segments)
