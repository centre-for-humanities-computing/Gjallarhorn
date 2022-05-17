from dejavu import Dejavu
from pathlib import Path
from dejavu.logic.recognizer.file_recognizer import FileRecognizer
import time
import os

import argparse

def main(args):
    """Fingerprint 1 minute snippets.

    Arguments:
        args {dict} -- dict specifying the channel name (r24syv|drp1) and the year(s) to create the index over
    """

    years = "_".join(args.years)

    db_name = args.channel + "_" +  years

    print(f"[INFO] Creating database {db_name}")
    os.system(f'sudo -u postgres psql -c "CREATE DATABASE {db_name};"')

    time.sleep(20)

    config = {
        "database": {
            "host": "127.0.0.1",
            "user": "postgres",
            "password": "newpass",
            "database": db_name,
            "port" : "5432"
        },
        "database_type": "postgres"
    }


    djv = Dejavu(config)

    BASE_DATA_PATH = Path("/work") / "data" / "p1-r24syv-dedup" 
    channel_path = args.channel if not args.channel == "r24syv" else "24syv"
    DATA_PATH = BASE_DATA_PATH / channel_path

    for year in DATA_PATH.iterdir():
        if year.name in args.years: 
            t0 = time.time()
            print(f"Start fingerprinting {year}..")
            djv.fingerprint_directory(str(year), [".wav"])
            print(f"Time taken: {time.time() - t0}")    

    # save to file
    t0 = time.time()
    os.system(f'pg_dump postgresql://postgres:newpass@127.0.0.1:5432/{db_name} > /work/data/p1-r24syv-dedup/index/{db_name}.psql')
    print(f"Time taken to dump db: {time.time() - t0}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--channel", type=str, help="channel, either drp1 or r24syv", required=True)
    parser.add_argument("-y", "--years", help="which years to iterate over", nargs="+", required=True)
    args = parser.parse_args()

    print(f"Processing {args.channel} over the years {args.years}")
    main(args)