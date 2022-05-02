""""Map files to metadata and identify reruns using pvica and doms data
TODO make it a class.."""

import glob
import os
import re
from functools import partial
from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

# import swifter
import psycopg2
from wasabi import msg

from dejavu import dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer
from utils import POSTGRES_INDEX_COMMANDS


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


class FingerprintDuplicateRemover:
    def __init__(self, index_path):
        self.index_path = Path(index_path)
        self.possible_indices = list(self.index_path.glob("*psql"))
        self.dejavu_db_config = {
            "database": {
                "host": "127.0.0.1",
                "user": "postgres",
                "password": "newpass",
                "database": "",
                "port": "5432",
            },
            "database_type": "postgres",
        }
        self.current_db = None
        self.is_indexed = False

    def _load_db(self, db_name: str, db_path: str) -> None:

        if db_name == self.current_db:
            msg.text(f"{db_name} already loaded. Continuing...")
            return
        # Create database
        os.system(f'sudo -u postgres psql -c "CREATE DATABASE {db_name};"')
        msg.info(f"Loading {db_name}...")

        ## Load database
        os.system(f"sudo -u postgres -i psql {db_name} < {str(db_path)}")
        self.current_db = db_name

    def _remove_db(self, db: str):
        # terminate all connections to the database
        os.system(
            f"sudo -u postgres psql -c 'SELECT pg_terminate_backend (pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = \"{db}\"';"
        )
        # drop database
        os.system(f"sudo -u postgres -c 'DROP DATABASE {db};")

        self.current_db = None
        self.is_indexed = False

    def _index_tables(self, db_name: str) -> None:
        if db_name == self.current_db and self.is_indexed:
            msg.text(f"{db_name} already indexed. Continuing...")
            return
        # Connect to database
        db_config = self.dejavu_db_config["database"]
        db_config["database"] = db_name
        conn = psycopg2.connect(*db_config)
        # do indexeing
        cur = conn.cursor()
        cur.execute(POSTGRES_INDEX_COMMANDS)
        cur.close()
        conn.close()

        self.is_indexed = True

    def _get_db_name(self, channel: str, year: Union[str, int]):
        regex_string = f"{channel}.*{str(year)}"
        matches = [re.search(regex_string, str(db)) for db in self.possible_indices]
        # Should only match a single index
        matched_idx = [i for i, match in enumerate(matches) if match]
        if len(matched_idx) > 1:
            raise ValueError(f"{regex_string} matched multiple indices")
        return self.possible_indices[matched_idx[0]]

    def _search_fingerprints(self, df: pd.DataFrame, db_name: str) -> pd.DataFrame:
        # setup dejavu
        config = self.dejavu_db_config
        config["database"]["database"] = db_name
        djv = Dejavu(self.dejavu_db_config)

        def dejavu_match(file_path: str) -> tuple:
            matches = djv.recognize(FileRecognizer, file_path)
            matched_files = [match["song_name"] for match in matches["results"]]
            confidences = [match["input_confidence"] for match in matches["results"]]

            return matched_files, confidences

        # df[["matched_files", "confidences"]] = zip(
        #     *df.swifter.allow_dask_on_strings(enable=True)["file_path"].apply(
        #         dejavu_match
        #     )
        # )
        df[["matched_files", "confidences"]] = zip(*df["file_path"].apply(dejavu_match))

    def find_duplicates(
        self,
        df: pd.DataFrame,
        channels: Union[List, str] = ["drp1", "r24syv"],
        years: Optional[Union[List[int], int]] = None,
    ) -> pd.DataFrame:
        """[summary]

        Arguments:
            df {pd.DataFrame} -- [description]

        Keyword Arguments:
            channels {Union[List, str]} -- [description] (default: {["drp1", "r24syv"]})
            years {Optional[Union[List[int], int]]} -- [description] (default: {None})

        Returns:
            pd.DataFrame -- [description]
        """

        if isinstance(channels, str):
            channels = [channels]
        if isinstance(years, int):
            years = [years]

        for channel in channels:
            for year in years:
                # Setup database stuff
                db_path = self._get_db_name(channel, year)
                db_name = db_path.stem
                self._load_db(db_name, db_path)
                self._index_tables(db_name)

                # Query database and get matched songs
                df = self._search_fingerprints(df, db_name)

                self._remove_db(db_name)

    def mark_duplicates(self, confidence_treshold):
        pass


if __name__ == "__main__":

    META_DIR = "/work/data/p1-r24syv/metadataFiles"
    RERUN_DIR = Path("/work") / "data" / "rerun_status" / "rerun_status"

    P1_FOLDERS = Path("/work/data/p1-r24syv/files/drp1")
    R24_FOLDERS = Path("/work/data/p1-r24syv/files/24syv")

    ### P1
    ## Check if files match metadata
    p1_filedict = get_filepath_dict(P1_FOLDERS)
    p1_metadata = get_metadata_dataframe("/drp1*")
    p1_metadata = add_matching_file_column(p1_metadata, p1_filedict)
    p1_n_na = p1_metadata["file_path"].isna().sum()

    print_file_statistics("p1", p1_filedict, p1_metadata, p1_n_na)

    ## Mark duplicates
    p1_metadata = mark_duplicates_description(p1_metadata)
    p1_metadata = add_rerun_info(p1_metadata)
    p1_metadata = merge_rerun_columns(p1_metadata)

    ### Radio 24syv
    r24_filedict = get_filepath_dict(R24_FOLDERS)
    r24_metadata = get_metadata_dataframe("/24syv*")
    r24_metadata = add_matching_file_column(r24_metadata, r24_filedict)
    r24_n_na = r24_metadata["file_path"].isna().sum()

    print_file_statistics("r24syv", r24_filedict, r24_metadata, r24_n_na)

    ## Mark duplicates
    r24_metadata = mark_duplicates_description(r24_metadata)
    r24_metadata = add_rerun_info(r24_metadata)
    r24_metadata = merge_rerun_columns(r24_metadata)

    detektor = p1_metadata[p1_metadata["dc:title"].str.contains("Detektor")]
    det_counts = detektor["episode"].value_counts().sort_values()
    ep_35 = detektor[detektor["episode"] == "35"]
    ### check deduplication on drp1: 2021-11

    ### exploratory stuff
    ## only keep 1 from each description with less than 10 counts
    desc_counts = p1_metadata["shortRecordDescription"].value_counts().reset_index()
    desc_counts = desc_counts.rename(
        {"index": "shortRecordDescription", "shortRecordDescription": "counts"},
        axis="columns",
    )
    p1_metadata = p1_metadata.merge(
        desc_counts, on="shortRecordDescription", how="left"
    )

    less_10 = p1_metadata[p1_metadata["counts"] < 10]
    less_10.drop_duplicates("shortRecordDescription")
