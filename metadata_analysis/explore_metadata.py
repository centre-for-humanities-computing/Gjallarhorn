""""Merge metadata files and save to a single csv for each channel"""

import glob
import os
import re
from functools import partial
from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from wasabi import msg

import time

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


def print_file_statistics(station: str, file_dict, metadata, n_na):
    print(
        f"""Number of {station} files: {len(file_dict)}
    Number of files in {station} metadata: {len(metadata)}
    Number of rows in metadata not matching a file: {n_na}"""
    )


def mark_duplicates_description(
    metadata: pd.DataFrame,
    remove_strings: List[str] = ["sendt første gang", "genudsendelse"],
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
    metadata["is_description_rerun"] = np.where(
        metadata["shortRecordDescription"].str.contains(regex_duplicate), True, False
    )
    print(
        f"{sum(metadata['is_description_rerun'])} duplicates found by searching for {remove_strings}"
    )
    return metadata


def mark_duplicates_title(
    metadata: pd.DataFrame,
    remove_strings: Union[List, str] =  "(G)" 
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
    metadata: pd.DataFrame,
    latest_time: int = 22,
    earliest_time: int = 6
):
    metadata["hour"] = pd.to_datetime(metadata["timestamp"])
    metadata["hour"] = metadata["hour"].dt.hour
    metadata["is_tod_rerun"] = np.where(
        (metadata["hour"] > latest_time) | (metadata["hour"] < earliest_time), True, False
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
    cols = [col for col in df.columns if "rerun" in col]

    df["is_rerun"] = False
    for col in cols:
        df["is_rerun"] = np.where(df[col] == True, True, df["is_rerun"])
    print(f"{sum(df['is_rerun'])} duplicates found in total")
    return df



def get_all_larm_links_from_matches(matches, confidences, decode=False):
    if decode:
        links = [get_larm_link(match.decode("utf-8")) for match in matches]
    else:
        links = [get_larm_link(match) for match in matches]
        

    for link, conf in zip(links, confidences):
        print(f"Link: {link} \n\tConfidence: {conf}")

def get_larm_link(pid: str):
    return f"https://larm.fm/#!object/id={pid}"


def get_all_larm_links_from_df(df, match_col = "matched_files_drp1_2014_2015", confidence_col = "confidences_drp1_2014_2015"):
     for row in df.iterrows():
        cur_file = Path(row[1]["filename"])
        print(f"Current file: {get_larm_link(cur_file.stem)}")
        get_all_larm_links_from_matches(row[1]["matched_files_drp1_2014_2015"], row[1]["confidences_drp1_2014_2015"])



if __name__ == "__main__":

    META_DIR = "/work/data/p1-r24syv/metadataFiles"
    RERUN_DIR = Path("/work") / "data" / "rerun_status" / "rerun_status"

    P1_FOLDERS = Path("/work/data/p1-r24syv/files/drp1")
    R24_FOLDERS = Path("/work/data/p1-r24syv/files/24syv")

    INDEX_PATH = Path("/work/data/p1-r24syv-dedup/index")

    ### P1
    ## Check if files match metadata
    # p1_filedict = get_filepath_dict(P1_FOLDERS)
    # p1_metadata = get_metadata_dataframe("/drp1*")
    # p1_metadata = add_matching_file_column(p1_metadata, p1_filedict)
    # p1_n_na = p1_metadata["file_path"].isna().sum()

    # print_file_statistics("p1", p1_filedict, p1_metadata, p1_n_na)

    # ## Mark duplicates
    # p1_metadata = mark_duplicates_description(p1_metadata)
    # # p1_metadata = add_rerun_info(p1_metadata)
    # p1_metadata = merge_rerun_columns(p1_metadata)

    # p1_metadata.to_csv()
    # # Test
    # test_data = p1_metadata[p1_metadata["year"] == "2014"].sample(100)
    #test_data.to_csv("/work/test_data.csv", index=False)
    # test_data = pd.read_csv("/work/49978/test_data.csv")
    #test_data = test_data.sample(5)
    # test_data = test_data.sample(5)

    # deduper = FingerprintDuplicateRemover(INDEX_PATH)
    # deduper.current_db = "drp1_2014_2015"
    # deduper.is_indexed = True
    # results = deduper.find_duplicates(test_data, channel="drp1", years=[2014])
    # res = results[["filename", "confidences_drp1_2014_2015", "matched_files_drp1_2014_2015", "is_rerun"]]

    # for row in res.iterrows():
    #     cur_file = Path(row[1]["filename"])
    #     print(f"Current file: {get_larm_link(cur_file.stem)}")
    #     get_all_larm_links_from_matches(row[1]["matched_files_drp1_2014_2015"], row[1]["confidences_drp1_2014_2015"])


    # test = "961c49c7-7a2a-4d16-b2f2-4fb86b17404d.mp3"
    # get_larm_description(test)
    # missing_rerun = p1_metadata[p1_metadata["filename"] == test]
    # missing_rerun = missing_rerun.reset_index(drop=True)
    # for col in missing_rerun.columns:
    #     print(col, "\t", missing_rerun[col][0])

    ### Radio 24syv
    # r24_filedict = get_filepath_dict(R24_FOLDERS)
    # r24_metadata = get_metadata_dataframe("/24syv*")
    # r24_metadata = add_matching_file_column(r24_metadata, r24_filedict)
    # r24_n_na = r24_metadata["file_path"].isna().sum()

    # print_file_statistics("r24syv", r24_filedict, r24_metadata, r24_n_na)

    # ## Mark duplicates
    # r24_metadata = mark_duplicates_description(r24_metadata)
    # r24_metadata = add_rerun_info(r24_metadata)
    # r24_metadata = merge_rerun_columns(r24_metadata)
    # r24_metadata.to_csv("r24syv_metadata.csv", index=False)

    # r24_2019_test = r24_metadata[r24_metadata["year"] == "2019"].sample(100)
    # r24_2019_test.to_csv("r24_test.csv", index=False)

    r24_metadata = pd.read_csv("/work/49978/Gjallarhorn/metadata_analysis/r24syv_metadata.csv")
    r24_2019_test = pd.read_csv("/work/49978/Gjallarhorn/metadata_analysis/r24_test.csv")
    deduper = FingerprintDuplicateRemover(INDEX_PATH)
    deduper.current_db = "r24syv_2019"
    deduper.is_indexed = True
    results = deduper.find_duplicates(r24_2019_test, channel="r24syv", years=[2019])
    res = results[["filename", "confidences_r24syv_2019", "matched_files_r24syv_2019", "is_rerun", "fileRef", "authID"]]

    # for some reason need to use fileRef to look up r24syv files on larm.fm


    def get_file_uid(match: str, metadata: pd.DataFrame):
        meta_match = metadata[metadata["filename"].str.contains(match)] 
        match_uid = meta_match["authID"].tolist()[0]
        # test if this works...
        match_uid = match_uid[-36:]
        #if not match_uid:
        #    match_uid = Path(meta_match["filename"].tolist()[0]).stem
        return match_uid
        


    for row in res.iterrows():
        cur_file = row[1]["authID"][-36:]
        if not cur_file:
            cur_file = Path(row[1]["filename"])
        matched_files = row[1]["matched_files_r24syv_2019"]
        matched_files = [m.decode("utf-8") for m in matched_files]
        safe_matches = [re.escape(m) for m in matched_files]
        matching_metadata = r24_metadata[r24_metadata["filename"].str.contains('|'.join(safe_matches))]
        matching_uids = [get_file_uid(m, matching_metadata) for m in safe_matches]

        print(f"Current file: {get_larm_link(cur_file)}")
        get_all_larm_links_from_matches(matching_uids, row[1]["confidences_r24syv_2019"], decode=False)


    # detektor = p1_metadata[p1_metadata["dc:title"].str.contains("Detektor")]
    # det_counts = detektor["episode"].value_counts().sort_values()
    # ep_35 = detektor[detektor["episode"] == "35"]
    # ### check deduplication on drp1: 2021-11

    # ### exploratory stuff
    # ## only keep 1 from each description with less than 10 counts
    # desc_counts = p1_metadata["shortRecordDescription"].value_counts().reset_index()
    # desc_counts = desc_counts.rename(
    #     {"index": "shortRecordDescription", "shortRecordDescription": "counts"},
    #     axis="columns",
    # )
    # p1_metadata = p1_metadata.merge(
    #     desc_counts, on="shortRecordDescription", how="left"
    # )

    # less_10 = p1_metadata[p1_metadata["counts"] < 10]
    # less_10.drop_duplicates("shortRecordDescription")
