import os

import soundfile as sf

from data_processing.convert_audiofile_to_segments import AudioConvert, Method


class DirectoryProcessor:

    def __init__(self, out_dir: str):
        self.out_dir = out_dir

        method = Method.SILERO
        self.audio_converter = AudioConvert(method=method)

    def process(self, directory: str):
        files = os.listdir(directory)
        for file in files:
            full_file_path = os.path.join(directory, file)
            segments = self.audio_converter.convert_file_to_segments(full_file_path)
            file_out_dir = self.create_subfolder_and_return_subfolder(file)
            self.segments_to_files(file_out_dir, segments)
            break

    def create_subfolder_and_return_subfolder(self, audio_id: str):
        subfolder = os.path.join(self.out_dir, audio_id[-4])
        os.mkdir(subfolder)
        return subfolder

    def segments_to_files(self, files_out_dir: str, audio_segments):
        for i, segment in enumerate(audio_segments):
            sf.write(f"{files_out_dir}/{i}.wav", segment[2], samplerate=16000)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Segment all audio files in a given directory')
    parser.add_argument("--directory_in", default="/work/2011-11", type=str)
    parser.add_argument("--directory_out", default="/work/processed_wav", type=str)
    args = parser.parse_args()

    dir_processor = DirectoryProcessor(out_dir=args.directory_out)
    dir_processor.process(args.directory_in)


