"""Messing around with audio fingerprinting"""

from typing import List
from audio_fingerprinting.query_fingerprint_index import FingerprintDuplicateRemover
from pathlib import Path

import pandas as pd

import re


def get_all_larm_links_from_matches(
    matches: List, confidences: List, decode=False
) -> None:
    """Print links to LARM.fm for the matched files

    Args:
        matches (List): Matched files
        confidences (List): Confidence of the match
        decode (bool, optional): Whether to decode from bytes to utf-8. Defaults to False.
    """
    if decode:
        links = [get_larm_link(match.decode("utf-8")) for match in matches]
    else:
        links = [get_larm_link(match) for match in matches]

    for link, conf in zip(links, confidences):
        print(f"Link: {link} \n\tConfidence: {conf}")


def get_larm_link(pid: str):
    return f"https://larm.fm/#!object/id={pid}"


def get_all_larm_links_from_df(
    df: pd.DataFrame,
    match_col: str,
    confidence_col: str,
) -> None:
    """Print links to LARM for all matches in a dataframe

    Args:
        df (pd.DataFrame): Dataframe with added rerun information
        match_col (str, optional): Which column the matches are stored in.
        confidence_col (str, optional): _description_.
    """
    for row in df.iterrows():
        cur_file = Path(row[1]["filename"])
        print(f"Current file: {get_larm_link(cur_file.stem)}")
        get_all_larm_links_from_matches(row[1][match_col], row[1][confidence_col])


def get_file_uid(match: str, metadata: pd.DataFrame) -> str:
    """Map a match to authID. Some files do not have the same authID as the
    file name as LARM uses as their index. This function converts the filename
    to the corresponding authID.


    Args:
        match (str): Name of the matching file
        metadata (pd.DataFrame): Metadatawith with 'filename' and 'authID' cols

    Returns:
        str: authID of the matching file
    """
    meta_match = metadata[metadata["filename"].str.contains(match)]
    match_uid = meta_match["authID"].tolist()[0]
    match_uid = match_uid[-36:]
    # if not match_uid:
    #    match_uid = Path(meta_match["filename"].tolist()[0]).stem
    return match_uid


if __name__ == "__main__":
    INDEX_PATH = Path("/work/data/p1-r24syv-dedup/index")
    METADATA_SAVE_DIR = Path("/work") / "data" / "p1-r24syv-dedup" / "metadata"

    ### P1
    p1_metadata = pd.read_csv(METADATA_SAVE_DIR / "p1_metadata_merged.csv")
    p1_test = p1_metadata[p1_metadata["year"] == "2014"].sample(100)
    p1_test.to_csv("p1_test.csv", index=False)

    deduper = FingerprintDuplicateRemover(INDEX_PATH)
    # deduper.current_db = "drp1_2014_2015"
    # deduper.is_indexed = True
    results = deduper.find_duplicates(p1_test, channel="drp1", years=[2014], debug=True)
    res = results[
        [
            "filename",
            "confidences_drp1_2014_2015",
            "matched_files_drp1_2014_2015",
            "is_rerun",
        ]
    ]

    for row in res.iterrows():
        cur_file = Path(row[1]["filename"])
        print(f"Current file: {get_larm_link(cur_file.stem)}")
        get_all_larm_links_from_matches(
            row[1]["matched_files_drp1_2014_2015"], row[1]["confidences_drp1_2014_2015"]
        )

    # Our description does not match the description on LARM for this file
    # "961c49c7-7a2a-4d16-b2f2-4fb86b17404d.mp3"

    ### R24syv
    r24_metadata = pd.read_csv(METADATA_SAVE_DIR / "r24syv_metadata_merged.csv")
    r24_2019_test = r24_metadata[r24_metadata["year"] == "2019"].sample(100)
    r24_2019_test.to_csv("r24_test.csv", index=False)

    deduper = FingerprintDuplicateRemover(INDEX_PATH)
    # deduper.current_db = "r24syv_2019"
    # deduper.is_indexed = True
    results = deduper.find_duplicates(r24_2019_test, channel="r24syv", years=[2019])
    res = results[
        [
            "filename",
            "confidences_r24syv_2019",
            "matched_files_r24syv_2019",
            "is_rerun",
            "fileRef",
            "authID",
        ]
    ]

    # for some reason need to use fileRef to look up r24syv files on larm.fm
    for row in res.iterrows():
        cur_file = row[1]["authID"][-36:]
        if not cur_file:
            cur_file = Path(row[1]["filename"])
        matched_files = row[1]["matched_files_r24syv_2019"]
        matched_files = [m.decode("utf-8") for m in matched_files]
        safe_matches = [re.escape(m) for m in matched_files]
        matching_metadata = r24_metadata[
            r24_metadata["filename"].str.contains("|".join(safe_matches))
        ]
        matching_uids = [get_file_uid(m, matching_metadata) for m in safe_matches]

        print(f"Current file: {get_larm_link(cur_file)}")
        get_all_larm_links_from_matches(
            matching_uids, row[1]["confidences_r24syv_2019"], decode=False
        )
