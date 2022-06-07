""""Merge metadata files and save to a single csv for each channel"""

import glob
from functools import partial
from pathlib import Path, PosixPath
from typing import Dict, List, Union

import numpy as np
import pandas as pd


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
    """Creates a dictionary mapping all files within subdirs of a directory (e.g. drp1)
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
    metadata = pd.concat(
        [metadata["metadata"].apply(pd.Series), metadata.drop("metadata", axis=1)],
        axis=1,
    )
    metadata = metadata.reset_index()
    metadata["shortRecordDescription"] = metadata["shortRecordDescription"].str.lower()
    return metadata


def add_matching_file_column(
    metadata_df: pd.DataFrame, file_dict: Dict
) -> pd.DataFrame:
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


def print_file_statistics(channel: str, file_dict, metadata, n_na):
    print(
        f"""Number of {channel} files: {len(file_dict)}
    Number of files in {channel} metadata: {len(metadata)}
    Number of rows in metadata not matching a file: {n_na}"""
    )


def mark_duplicates_description(
    metadata: pd.DataFrame,
    remove_strings: List[str] = ["sendt første gang", "genudsendelse"],
) -> pd.DataFrame:
    """Remove files/rows that contain any of the phrases in `remove_strings`.
    Only works for drp1 (no description for r24syv)

    Arguments:
        metadata {pd.DataFrame} -- Datafrane with metadata
        remove_strings {List[str]} (default: {["sendt første gang", "genudsendelse"]}) --
            list of strings to search for an remove

    Returns:
        pd.DataFrame
    """
    regex_duplicate = "|".join(remove_strings)
    metadata["is_description_rerun"] = np.where(
        metadata["shortRecordDescription"].str.contains(regex_duplicate), True, False
    )
    print(
        f"{sum(metadata['is_description_rerun'])} duplicates found by searching for {remove_strings}"
    )
    return metadata


def mark_duplicates_title(
    metadata: pd.DataFrame, remove_strings: Union[List, str] = "(G)"
):
    """Mark duplicates based on title (especially useful for r24syv)

    Arguments:
        metadata {pd.DataFrame} -- Metadata dataframe

    Keyword Arguments:
        remove_strings {Union[List, str]} -- strings to search for (default: {"(G)"})
    """
    if isinstance(remove_strings, list):
        remove_strings = "|".join(remove_strings)
    metadata["is_title_rerun"] = np.where(
        metadata["dc:title"].str.contains(remove_strings), True, False
    )
    print(
        f"{sum(metadata['is_title_rerun'])} duplicates found by searching for {remove_strings} in title"
    )
    return metadata


def mark_duplicates_time_of_day(
    metadata: pd.DataFrame, latest_time: int = 22, earliest_time: int = 6
) -> pd.DataFrame:
    """Mark duplicates by time of day of airing.

    Args:
        metadata (pd.DataFrame): Metadata dataframe
        latest_time (int, optional): Latest hour to include. Defaults to 22.
        earliest_time (int, optional): Earliest hour to include. Defaults to 6.

    Returns:
        pd.DataFrame: Dataframe with `is_tod_rerun` column
    """
    metadata["hour"] = pd.to_datetime(metadata["timestamp"])
    metadata["hour"] = metadata["hour"].dt.hour
    metadata["is_tod_rerun"] = np.where(
        (metadata["hour"] > latest_time) | (metadata["hour"] < earliest_time),
        True,
        False,
    )
    print(
        f"{sum(metadata['is_tod_rerun'])} potential duplicates found by restricting time of day to be be later than {earliest_time} and earlier than {latest_time}"
    )
    return metadata


def add_rerun_info(df: pd.DataFrame):
    """Add `doms_rerun` and `pvica_rerun` columns indicating if the show is a rerun"""
    doms = pd.read_csv(RERUN_DIR / "doms.csv", sep="\t", header=None)
    pvica = pd.read_csv(
        RERUN_DIR / "pvica-20220323.csv", sep="\t", header=None, skiprows=1
    )
    doms.columns = ["authID", "is_doms_rerun"]
    pvica.columns = ["authID", "is_pvica_rerun"]

    df = df.merge(pvica, how="left", on="authID")
    df = df.merge(doms, how="left", on="authID")

    df["is_pvica_rerun"] = np.where(df["is_pvica_rerun"] == "rerun", True, False)
    df["is_doms_rerun"] = np.where(df["is_doms_rerun"] == "rerun", True, False)
    return df


def merge_rerun_columns(df: pd.DataFrame):
    """Merge all columns containing the substring `rerun`. If any of them is
    True, `is_rerun` will be True

    Args:
        df (pd.DataFrame): Metadata dataframe

    Returns:
        _type_: Dataframe with added `is_rerun` column
    """
    cols = [col for col in df.columns if "rerun" in col]

    df["is_rerun"] = False
    for col in cols:
        df["is_rerun"] = np.where(df[col] == True, True, df["is_rerun"])
    print(f"{sum(df['is_rerun'])} duplicates found in total")
    return df


if __name__ == "__main__":

    META_DIR = "/work/data/p1-r24syv/metadataFiles"
    RERUN_DIR = Path("/work") / "data" / "rerun_status" / "rerun_status"

    P1_FOLDERS = Path("/work/data/p1-r24syv/files/drp1")
    R24_FOLDERS = Path("/work/data/p1-r24syv/files/24syv")

    SAVE_FOLDER = Path("/work") / "data" / "p1-r24syv-dedup" / "metadata"
    if not SAVE_FOLDER.exists():
        SAVE_FOLDER.mkdir()

    ## P1
    # Check if files match metadata
    p1_filedict = get_filepath_dict(P1_FOLDERS)
    p1_metadata = get_metadata_dataframe("/drp1*")
    p1_metadata = add_matching_file_column(p1_metadata, p1_filedict)
    p1_n_na = p1_metadata["file_path"].isna().sum()

    print_file_statistics("p1", p1_filedict, p1_metadata, p1_n_na)

    ## Mark duplicates by description and time of day
    p1_metadata = mark_duplicates_description(p1_metadata)
    p1_metadata = mark_duplicates_time_of_day(p1_metadata)
    p1_metadata = add_rerun_info(p1_metadata)
    p1_metadata = merge_rerun_columns(p1_metadata)

    p1_metadata.to_csv(SAVE_FOLDER / "p1_metadata_merged.csv", index=False)
    p1_no_reruns = p1_metadata[p1_metadata["is_rerun"] == False]
    # only keep entries with a matching file
    p1_no_reruns = p1_no_reruns[~(p1_no_reruns["file_path"].isnull())]
    print(f"Number of files left: {p1_no_reruns.shape[0]}")
    p1_no_reruns["file_path"].to_csv(SAVE_FOLDER / "p1_no_reruns.txt", index=False, header=False)

    ## Radio 24syv
    r24_filedict = get_filepath_dict(R24_FOLDERS)
    r24_metadata = get_metadata_dataframe("/24syv*")
    r24_metadata = add_matching_file_column(r24_metadata, r24_filedict)
    r24_n_na = r24_metadata["file_path"].isna().sum()

    print_file_statistics("r24syv", r24_filedict, r24_metadata, r24_n_na)

    ## Mark duplicates by title and time of day (no descriptions for r24syv)
    r24_metadata = mark_duplicates_title(r24_metadata)
    r24_metadata = mark_duplicates_time_of_day(r24_metadata)
    r24_metadata = add_rerun_info(r24_metadata)
    r24_metadata = merge_rerun_columns(r24_metadata)
    r24_metadata.to_csv(SAVE_FOLDER / "r24syv_metadata_merged.csv", index=False)
    r24_no_reruns = r24_metadata[r24_metadata["is_rerun"] == False]
    r24_no_reruns = r24_no_reruns[~(r24_no_reruns["file_path"].isnull())]

    print(f"Number of files left: {r24_no_reruns.shape[0]}")
    r24_no_reruns["file_path"].to_csv(SAVE_FOLDER / "r24syv_no_reruns.txt", index=False, header=False)
