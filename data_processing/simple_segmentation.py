
import numpy as np


class SimpleSegmentation:

    def __init__(self):
        self.max_length_seconds = 15  # Seconds
        self.min_length_seconds = 3  # Seconds
        self.max_length_frames = self.max_length_seconds * 16000
        self.min_length_frames = self.min_length_seconds * 16000
        self.always_crop_frames_seconds = 60
        self.always_crop_frames = self.always_crop_frames_seconds * 16000

    def segment(self, audio: np.array):

        audio = audio[self.always_crop_frames:-self.always_crop_frames]

        frame_iterator = 0
        interval_iterator = 0
        intervals = np.random.random_integers(self.min_length_frames, self.max_length_frames, len(audio) // 5)

        if len(intervals) == 0:
            print("Ignoring due to too short file...")
            return []

        all_segments = []
        while True:
            sound_length = intervals[interval_iterator]

            # Break when reaching end.
            if frame_iterator + sound_length > len(audio):
                break

            temp_sound = audio[frame_iterator:frame_iterator + sound_length]
            all_segments.append((frame_iterator, frame_iterator + sound_length, temp_sound))
            frame_iterator += sound_length
            interval_iterator += 1

        return all_segments

if __name__ == '__main__':
    import ffmpeg

    def read_file_to_np(audiofile_path: str):
        out, err = (
            ffmpeg
                .input(audiofile_path)
                .output('pipe:', format="wav", acodec="pcm_s16le", ar=16000, ac=1)
                .run(capture_stdout=True)
        )
        numpy_array = np.frombuffer(out, dtype=np.int16)
        return numpy_array


    file = "/home/rafje/Downloads/Foelg-pengene--Apple--_823566a09c664d17aad77862d288473a_192.mp3"
    audio = read_file_to_np(file)
    simple_segmentation = SimpleSegmentation()
    segments = simple_segmentation.segment(audio)
    print(segments[0])
    print(len(segments))
    import soundfile as sf
    sf.write("test.wav", segments[0][2], samplerate=16000)
