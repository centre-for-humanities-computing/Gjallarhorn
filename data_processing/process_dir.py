import os

import soundfile as sf

from data_processing.convert_audiofile_to_segments import AudioConvert, Method


class DirectoryProcessor:

    def __init__(self, out_dir: str, use_gpu: bool = False):
        self.out_dir = out_dir

        method = Method.SILERO
        self.audio_converter = AudioConvert(method=method, use_gpu=use_gpu)

    def process(self, directory: str):
        files = os.listdir(directory)
        for i, file in enumerate(files):
            print(f"Processing file: {file}")
            full_file_path = os.path.join(directory, file)

            file_out_dir = self.create_subfolder_and_return_subfolder(file)
            if not file_out_dir:
                print(f"File was already processed: {file}")
                print("Continueing to next")
                continue

            print("Loading and segmenting...")
            segments = self.audio_converter.convert_file_to_segments(full_file_path)

            print(f"Saving to segments to: {full_file_path}")
            self.segments_to_files(file_out_dir, segments)
            if i % 10 == 0:
                print(f"progress: {i}/{len(files)}")

    def create_subfolder_and_return_subfolder(self, audio_id: str):
        subfolder = os.path.join(self.out_dir, audio_id[:-4])
        if os.path.exists(subfolder):
            return None

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
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    dir_processor = DirectoryProcessor(out_dir=args.directory_out, use_gpu=args.use_gpu)
    dir_processor.process(args.directory_in)
