""""Methods to calculate the cross correlation / lag between two audio signals. 
Used to identify duplicate files"""
import os
from pathlib import Path, PosixPath
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import scipy
import torchaudio
from scipy.signal import correlate


class DuplicateChecker:

    def __init__(self, frame_offset: int = 60*44100, num_frames: int = 2*6*44100, cor_threshold: float=100):
        self.frame_offset = frame_offset
        self.num_frames = num_frames
        self.cor_threshold = cor_threshold

    def load_file(self, path):
        # Discard first frame_offset frames and load num_frames
        sig, sr = torchaudio.load(path, frame_offset=self.frame_offset, num_frames=self.num_frames)
        return sig.numpy()

    def correlate_files(self, paths: List[str]):
        file_dict = {path : self.load_file(path) for path in paths}

        return file_dict


def get_full_file_name(filename: str, all_files: Dict[str, str]):
    """Return the full path of a file if it exists. Else, return np.nan

    Arguments:
        filename {str} -- filename/programid of the file
        all_files {Dict[str, str]} -- dict mapping filename to full filepath
    """
    try:
        path = all_files[filename]
        return path
    except KeyError:
        return np.nan


def get_filepath_dict(folder_path: PosixPath) -> Dict:
    """Creates a dictionary mapping all files within subdirs of a directory (e.g. drp1 fileg)
    to their programID

    Arguments:
        folder_path {PosixPath} -- a Path object
    Returns Dict with program id as key and full filename as value
    """
    files = [[p for p in folder.iterdir()] for folder in folder_path.iterdir()]
    files = [item for sublist in files for item in sublist]
    program_ids = [f.name for f in files]

    return {p_id: f for p_id, f in zip(program_ids, files)}


def get_metadata_dataframe(glob_pattern: str) -> pd.DataFrame:
    """Loads a dataframe of metadata based on a glob pattern (e.g. /drp1*, /24syv*)

    Arguments:
        glob_pattern {str}

    Returns:
        pd.DataFrame
    """
    metadata_files = glob.glob(META_DIR + glob_pattern)
    metadata = pd.concat([pd.read_json(file, lines=True) for file in metadata_files])
    # unnest metadata column
    metadata = pd.concat([metadata["metadata"].apply(pd.Series), metadata.drop('metadata', axis=1)], axis=1)
    metadata = metadata.reset_index()
    metadata["shortRecordDescription"] = metadata["shortRecordDescription"].str.lower()
    return metadata


def add_matching_file_column(metadata_df: pd.DataFrame, file_dict: Dict) -> pd.DataFrame:
    """Add a column to a metadata df with the full filename of the file.
    If no match, np.nan is added. 

    Arguments:
        metadata_df {pd.DataFrame} -- Metadata dataframe 
        file_dict {Dict} -- dictionary of filename and filepaths

    Returns:
        pd.DataFrame
    """
    get_filename = partial(get_full_file_name, all_files=file_dict)
    metadata_df["file_path"] = metadata_df["filename"].apply(get_filename)

    return metadata_df


def print_file_statistics(station: str, file_dict, metadata, n_na):
    print(f"""Number of {station} files: {len(file_dict)}
    Number of files in {station} metadata: {len(metadata)}
    Number of rows in metadata not matching a file: {n_na}""")


def remove_duplicates_description(
    metadata: pd.DataFrame, 
    remove_strings: List[str] = ["sendt første gang", "genudsendelse"]
    ) -> pd.DataFrame:
    """Remove files/rows that contain any of the phrases in `remove_strings`

    Arguments:
        metadata {pd.DataFrame} -- 
        remove_strings {List[str]} (default: {["sendt første gang", "genudsendelse"]}) -- 
            list of strings to search for an remove

    Returns:
        pd.DataFrame
    """
    regex_duplicate = "|".join(remove_strings)
    duplicates = metadata[metadata["shortRecordDescription"].str.contains(regex_duplicate)]
    print(f"{len(duplicates)} duplicates found by searching for {remove_strings}")
    return metadata.drop(duplicates.index)




if __name__ == "__main__":
    import pandas as pd
    import os
    from pathlib import Path
    import glob
    from functools import partial

    META_DIR = "/work/p1-r24syv/metadataFiles"
    
    P1_FOLDERS = Path("/work/p1-r24syv/files/drp1")
    R24_FOLDERS = Path("/work/p1-r24syv/files/24syv")

    ## Check if files match metadata
    p1_filedict = get_filepath_dict(P1_FOLDERS)
    p1_metadata = get_metadata_dataframe("/drp1*")
    p1_metadata = add_matching_file_column(p1_metadata, p1_filedict)
    p1_n_na = p1_metadata["file_path"].isna().sum()

    print_file_statistics("p1", p1_filedict, p1_metadata, p1_n_na)

    r24_filedict = get_filepath_dict(R24_FOLDERS)
    r24_metadata = get_metadata_dataframe("/24syv*")
    r24_metadata = add_matching_file_column(r24_metadata, r24_filedict)
    r24_n_na = r24_metadata["file_path"].isna().sum()

    print_file_statistics("r24syv", r24_filedict, r24_metadata, r24_n_na)

    remove_duplicates_description(p1_metadata)


    detektor = p1_metadata[p1_metadata["dc:title"].str.contains("Detektor")]
    det_counts = detektor["episode"].value_counts().sort_values()
    ep_35 = detektor[detektor["episode"] == "35"]
    ### check deduplication on drp1: 2021-11

    ### exploratory stuff
    ## only keep 1 from each description with less than 10 counts
    desc_counts = p1_metadata["shortRecordDescription"].value_counts().reset_index()
    desc_counts = desc_counts.rename({"index" : "shortRecordDescription", "shortRecordDescription" : "counts"}, axis="columns")
    p1_metadata = p1_metadata.merge(desc_counts, on="shortRecordDescription", how="left")

    less_10 = p1_metadata[p1_metadata["counts"] < 10]
    less_10.drop_duplicates("shortRecordDescription")