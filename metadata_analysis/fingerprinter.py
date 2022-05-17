
"""Class for handling postgresql databases and querying audio files with Dejavu"""
import os
import re
from pathlib import Path, 
from typing import Dict, List, Optional, Set, Union

import pandas as pd

import psycopg2
from wasabi import msg

from dejavu import Dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer
from utils import POSTGRES_INDEX_COMMANDS_LIST

from multiprocess import Pool

import time

class FingerprintDuplicateRemover:
    def __init__(self, index_path: str, base_snippet_path: str = "/work/data/p1-r24syv-dedup/ten_sec_snippets/"):
        self.index_path = Path(index_path)
        self.base_snippet_path = base_snippet_path
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
        """Loads a postgresql database from a .sql dump

        Args:
            db_name (str): Name to load database to
            db_path (str): Path to the .sql dump
        """        

        if db_name == self.current_db:
            msg.text(f"{db_name} already loaded. Continuing...")
            return
        # Create database
        os.system(f'sudo -u postgres psql -c "CREATE DATABASE {db_name};"')

        ## Load database
        with msg.loading(f"Loading {db_name}..."):
            os.system(f"sudo -u postgres -i psql {db_name} < {str(db_path)}")
        msg.good(f"{db_name} loaded!")
        self.current_db = db_name

    def _remove_db(self, db_name: str):
        """Remove a postgres database to save space and memory.

        Args:
            db_name (str): Name of the database to remove
        """
        # terminate all connections to the database
        msg.info(f"Removing {db_name}")
        os.system(
            f"sudo -u postgres psql -c 'SELECT pg_terminate_backend (pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = \"{db_name}\"';"
        )
        # drop database
        os.system(f"sudo -u postgres -c 'DROP DATABASE {db_name};")

        self.current_db = None
        self.is_indexed = False

    def _index_tables(self, db_name: str) -> None:
        """Index all the tables in a given database for faster queries

        Args:
            db_name (str): Database to index 
        """
        if db_name == self.current_db and self.is_indexed:
            msg.text(f"{db_name} already indexed. Continuing...")
            return
        # Connect to database
        db_config = self.dejavu_db_config["database"]
        db_config["database"] = db_name
        conn = psycopg2.connect(dbname=db_name, user=db_config["user"], password=db_config["password"], host=db_config["host"])
        # do indexing
        cur = conn.cursor()
        with msg.loading(f"Indexing {db_name}..."):
            for index_command in POSTGRES_INDEX_COMMANDS_LIST:
                msg.info(f"Running {index_command}")
                cur.execute(index_command)
                conn.commit()
        msg.good(f"{db_name} indexed!")
        cur.close()
        conn.close()
        self.is_indexed = True

    def _get_db_name(self, channel: str, year: Union[str, int]) -> Path:
        """Get the name of the database based on a channel and year

        Args:
            channel (str): Which channel the index is for (r24syv drp1)
            year (Union[str, int]): Which year the index spans

        Raises:
            ValueError: If multiple databases match the name (sanity check)

        Returns:
            str: Path to the database
        """        
        regex_string = f"{channel}_.*{str(year)}"
        matches = [re.search(regex_string, str(db)) for db in self.possible_indices]
        # Should only match a single index
        matched_idx = [i for i, match in enumerate(matches) if match]
        if len(matched_idx) > 1:
            raise ValueError(f"{regex_string} matched multiple indices")
        return self.possible_indices[matched_idx[0]]

    def _add_snippet_paths(self, df: pd.DataFrame, channel: str) -> pd.DataFrame:
        """Add the path to the 10 ten second snippets to a metadata dataframe

        Args:
            df (pd.DataFrame): Metadata dataframe
            channel (str): Either drp1 or r24syv

        Returns:
            pd.DataFrame: Dataframe with added column 'snippet_path'  
        """        
        df["snippet_path"] = self.base_snippet_path + channel + "/" + df["year"].astype(str) + "/" + df["filename"]
        # remove suffix (not completely systematic, sometimes 
        # .mp3, sometimes no suffix))
        df["snippet_path"] = df["snippet_path"].str.replace("\.(mp3|wav)$", "", regex=True)
        df["snippet_path"] = df["snippet_path"] + ".wav"
        return df


    def _search_fingerprints(self, df: pd.DataFrame, db_name: str) -> pd.DataFrame:
        """Search for matching fingerprints in a database

        Args:
            df (pd.DataFrame): Dataframe with 'snippet_path' column
            db_name (str): Database to search for matches in

        Returns:
            pd.DataFrame: _description_
        """        
        # setup dejavu
        config = self.dejavu_db_config
        config["database"]["database"] = db_name

        def dejavu_match(file_path: str) -> tuple:
            djv = Dejavu(config)
            error = False
            try:
                matches = djv.recognize(FileRecognizer, file_path)
                matched_files = [match["song_name"] for match in matches["results"]]
                confidences = [match["input_confidence"] for match in matches["results"]]
            except:
                error = True
                matched_files, confidences = [], []

            return matched_files, confidences, error

        pool = Pool()
        paths = df["snippet_path"].to_list()
        
        t0 = time.time()
        results = pool.map(dejavu_match, paths)
        print(f"Time taken: {time.time() - t0}")

        matches, confidences, errors = zip(*results)
        msg.good("Done matching!")
        pool.close()
        pool.join()
        
        out_df = pd.DataFrame(
            {f"matched_files_{db_name}" : [None] * len(df), 
            f"confidences_{db_name}" : [None] * len(df),
            f"error_{db_name}" : [None] * len(df)})

        out_df[f"matched_files_{db_name}"] = matches
        out_df[f"confidences_{db_name}"] = confidences
        out_df[f"error_{db_name}"] = errors
        out_df.index = df.index

        n_errors = out_df[f"error_{db_name}"].sum()
        msg.warn(f"{n_errors} files not found!")
        return out_df

    def find_duplicates(
        self,
        df: pd.DataFrame,
        channel: str,
        years: Optional[Union[List[int], int]],
        debug: bool = False
    ) -> pd.DataFrame:
        """Search for duplicates in all files in a dataframe for either r24syv or drp1.
        Search in a database containing just one year or iterate over multiple years.
        Main user-facing function. 

        Args:
            df (pd.DataFrame): Dataframe with metadata (filename)
            channel (str): Which channel to search for duplicates in (r24syv or drp1)
            years (Optional[Union[List[int], int]], optional): Which years to search for
            debug (bool): Whether to remove the loaded database after querying

        Returns:
            pd.DataFrame: Metadataframe with added columns "[matched_files|confidences|erorr]_{dbname}" 
        """

        if isinstance(years, int):
            years = [years]
    
        matches = []
        df = self._add_snippet_paths(df, channel)

        for year in years:
            # Setup database stuff
            db_path = self._get_db_name(channel, year)
            db_name = db_path.stem
            self._load_db(db_name, db_path)
            self._index_tables(db_name)
            
            # Query database and get matched songs
            matches.append(self._search_fingerprints(df, db_name))
            if not debug:
                self._remove_db(db_name)
        matches = pd.concat(matches, axis=1)
        return pd.concat([df, matches], axis=1)

    def mark_duplicates(self, confidence_treshold):
        pass

if __name__ == "__main__":
    # # example use case   
    # INDEX_PATH = Path("/work/data/p1-r24syv-dedup/index")
    # metadata = pd.load("path_to_metadata.csv")

    # deduper = FingerprintDuplicateRemover(INDEX_PATH)
    # results = deduper.find_duplicates(test_data, channel="drp1", years=[2014])
    # res = results[["filename", "confidences_drp1_2014_2015", "matched_files_drp1_2014_2015", "is_rerun"]]