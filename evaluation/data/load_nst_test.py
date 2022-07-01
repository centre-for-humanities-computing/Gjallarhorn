import pandas as pd
from pathlib import Path
from wasabi import msg

from typing import Union

from datasets import load_dataset

NST_DIR = Path("/work/data/speech-finetuning/nst")

def load_nst_data():
    msg.info("Loading NST test set...")
    df = pd.read_csv(NST_DIR / "preprocessed_test" / "data_info.csv")
    df["filepath"] = df["file"].apply(lambda x: NST_DIR / "preprocessed_test" / x)
    msg.good("NST test set loaded!")
    return df["filepath"].tolist(), df["trans"].tolist()

