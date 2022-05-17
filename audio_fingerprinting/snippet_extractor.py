"""Class for extracting short segments from .wav files. 
Used for constructing the index of audio fingerprints as well as querying it.

If called as main will extract 1 minute and 10 second snippets from r24syv and drp1.
"""
import os
import time
from multiprocessing import Pool
from pathlib import Path, PosixPath
from typing import List, Union, Iterable
import numpy as np


class SnippetExtractor():
    def __init__(self, start_s: int, duration: int, overwrite_dir: bool = False, overwrite_files: bool = False):
        """Extract short audio segments from individual files or a directory. In files can be any type, but will output to .wav.

        Args:
            start_s (int): How many seconds into the file the segment should start
            duration (int): How many seconds to extracts
            overwrite_dir (bool, optional): Whether to overwrite the save dir if it already exists. Defaults to False.
            overwrite_files (bool, optional): Whether to overwrite files if they already exist. Defaults to False.
        """
        self.start_s = start_s
        self.duration = duration
        self.overwrite_dir = overwrite_dir
        self.overwrite_files = "-y" if overwrite_files else "-n"

    @staticmethod
    def mp3_to_wav_name(path: Path) -> Path:
        """Change file suffix to .wav."""
        return path.with_suffix(".wav")


    @staticmethod
    def ffmpeg_extract_snippet_from_file(in_path: str, out_path: Path, start: int, duration: int, overwrite_file: str) -> None:
        """Call ffmpeg from command line to extract a snippet from an audio file and convert it to wav.

        Arguments:
            in_path {str} -- path of the file to extract from
            out_path {Path} -- path to write the chunk to
            start {int} -- how many seconds in to the file to start the chunk. 
            duration {int} -- how long the chunk should be (in seconds). 
        """
        os.system(f"ffmpeg {overwrite_file} -hide_banner -loglevel error -ss {start} -i {in_path} -t {duration} {SnippetExtractor.mp3_to_wav_name(out_path)}")


    def extract_snippets_from_dirs(self, in_dir: Union[Iterable[Path], Path], out_dir: Path) -> None:
        """Extract sound segments from all files from a list of folders.

        Args:
            in_dir (Union[Iterable[Path], Path]): The directories to parse
            out_dir (Path): Where to save the extracted files

        Returns:
            _type_: _description_
        """
        if out_dir.exists() and not self.overwrite_dir:
            print(f"Directory {out_dir} already exists. Skipping..")
            return None
        if not isinstance(in_dir, PosixPath):
            for i_dir in in_dir:
                self.extract_snippets_from_dir(i_dir, out_dir)
        else:
            self.extract_snippets_from_dir(in_dir, out_dir)

        
    def extract_snippets_from_dir(self, in_dir: Path, out_dir: Path) -> None:
        """Extract sound segments from all files in a directory

        Args:
            in_dir (Path): The directory to parse
            out_dir (Path): Where to save the extracted files
        """
        self.out_dir = out_dir

        # if out_dir.exists() and not self.overwrite_dir:
        #     print(f"Directory {out_dir} already exists. Skipping..")
        #     return None

        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        
        pool = Pool()
        print(f"[INFO] Extracting snippets from {in_dir}...")
        t0 = time.time()
        pool.map(self._wrap_ffmpeg_extract_snippet, in_dir.iterdir())
        print(f"Time taken: {time.time() - t0}")
        pool.close()
        pool.join()

    
    def _wrap_ffmpeg_extract_snippet(self, file_path: Path) -> None:
        """"Wrapper for ffmpeg_extract_snippet_from_file to make it `Pool`able"""
        out_path = self.out_dir / file_path.name
        self.ffmpeg_extract_snippet_from_file(in_path=file_path, out_path=out_path, start=self.start_s, duration=self.duration, overwrite_file=self.overwrite_files)


if __name__ == "__main__":
    P1_DIRS = Path("/work") / "data" / "p1-r24syv" / "files" / "drp1"
    BASE_OUT_PATH = Path("/work") / "data" / "p1-r24syv-dedup"
    p1_years = [str(year) for year in np.arange(2005, 2022)]

    min_snipper = SnippetExtractor(start_s=180, duration=60)
    ten_sec_snipper = SnippetExtractor(start_s=205, duration=10)


    print("[INFO] Extracting from P1..")
    for year in p1_years:
        print(f"[INFO] Starting year {year}...")
        t0 = time.time()
        min_out_dir = BASE_OUT_PATH / "drp1" / year
        ten_sec_out_dir = BASE_OUT_PATH / "ten_sec_snippets" / "drp1" / year
        in_dirs = list(P1_DIRS.glob(year + "*"))

        print("[INFO] Extracting 1 min snippets...")
        min_snipper.extract_snippets_from_dirs(in_dirs, min_out_dir)
        print("[INFO] Extracting 10 second snippets...")
        ten_sec_snipper.extract_snippets_from_dirs(in_dirs, ten_sec_out_dir)
        print(f"[INFO] Time taken for year {year}: {time.time() - t0}")

    R24_DIRS = Path("/work") / "data" / "p1-r24syv" / "files" / "24syv"
    r24_years = [str(year) for year in np.arange(2011, 2020)]

    print("[INFO] Extracting from R24syv..")
    for year in r24_years:
        print(f"[INFO] Starting year {year}...")
        t0 = time.time()
        min_out_dir = BASE_OUT_PATH / "r24syv" / year
        ten_sec_out_dir = BASE_OUT_PATH / "ten_sec_snippets" / "r24syv" / year
        in_dirs = list(R24_DIRS.glob(year + "*"))
        print("[INFO] Extracting 1 minute snippets...")
        min_snipper.extract_snippets_from_dirs(in_dirs, min_out_dir)
        print("[INFO] Extracting 10 second snippets...")
        ten_sec_snipper.extract_snippets_from_dirs(in_dirs, ten_sec_out_dir)
        print(f"[INFO] Time taken for year {year}: {time.time() - t0}")


    


