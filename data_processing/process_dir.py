import os
from typing import List, Tuple

import numpy as np
import soundfile as sf

from data_processing.convert_audiofile_to_segments import AudioConvert, Method


class DirectoryProcessor:

    def __init__(self, out_dir: str, use_gpu: bool = False):
        self.out_dir = out_dir

        method = Method.SIMPLE
        self.audio_converter = AudioConvert(method=method, use_gpu=use_gpu)

        duplicates_list = self.load_all_dedupped_files()
        self.dedup_lookup_dict = {k: 1 for k in duplicates_list}

    def load_all_dedupped_files(self):
        dedup_files = ["/work/data/p1-r24syv-dedup/metadata/p1_no_reruns.txt",
                       "/work/data/p1-r24syv-dedup/metadata/r24syv_no_reruns.txt"]

        all_duplicates = []
        for dedup_file in dedup_files:
            all_duplicates += self.load_dedup_file_list(dedup_file)

        return all_duplicates

    @staticmethod
    def load_dedup_file_list(file_path: str):
        all_uuids_of_duplicates = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
            for line in data:
                specific_uuid = line.split("/")[-1]
                all_uuids_of_duplicates.append(specific_uuid)

        return all_uuids_of_duplicates

    def process(self, directory: str):
        mp3_files = []
        for root, subdirs, files in os.walk(directory):
            for f in files:
                if f in self.dedup_lookup_dict:
                    mp3_files.append((root, f))

        for i, (root, file) in enumerate(mp3_files):
            print(f"Processing file: {file}")
            full_file_path = os.path.join(root, file)

            file_out_dir = self.create_subfolder_and_return_subfolder(file)
            if not file_out_dir:
                print(f"File was already processed: {file}")
                print("Continueing to next")
                continue

            print("Loading and segmenting...")
            segments = self.audio_converter.convert_file_to_segments(full_file_path)

            print(f"Saving to segments to: {file_out_dir}")
            self.segments_to_files(file_out_dir, segments)
            self.write_tsv_format(file_out_dir, segments)
            if i % 10 == 0:
                print(f"progress: {i}/{len(mp3_files)}")

    def write_tsv_format(self, files_out_dir: str, audio_segments: List[Tuple[int, int, np.array]]):
        write_strings_list = []
        for i, segment in enumerate(audio_segments):
            write_strings_list.append(f"{i}.wav\t{len(segment[2])}")

        with open(os.path.join(files_out_dir, "metadata.tsv"), "w", encoding="utf-8") as f:
            f.write("\n".join(write_strings_list))

    def create_subfolder_and_return_subfolder(self, audio_id: str):
        subfolder = os.path.join(self.out_dir, audio_id[:-4])
        if os.path.exists(subfolder):
            return None

        os.mkdir(subfolder)
        return subfolder

    def segments_to_files(self, files_out_dir: str, audio_segments: List[Tuple[int, int, np.array]]):
        for i, segment in enumerate(audio_segments):
            sf.write(f"{files_out_dir}/{i}.wav", segment[2].numpy().astype(np.int16), samplerate=16000)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Segment all audio files in a given directory')
    parser.add_argument("--directory_in", default="/work/2011-11", type=str)
    parser.add_argument("--directory_out", default="/work/processed_wav", type=str)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    import time

    start = time.time()
    dir_processor = DirectoryProcessor(out_dir=args.directory_out, use_gpu=args.use_gpu)
    dir_processor.process(args.directory_in)
    print(time.time() - start)
