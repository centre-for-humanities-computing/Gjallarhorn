import pandas as pd

from pathlib import Path
import os

from num2words import num2words

vocab = {
    "<pad>": 0,
    "<unk>": 1,
    " ": 2,
    "a": 3,
    "b": 4,
    "c": 5,
    "d": 6,
    "e": 7,
    "f": 8,
    "g": 9,
    "h": 10,
    "i": 11,
    "j": 12,
    "k": 13,
    "l": 14,
    "m": 15,
    "n": 16,
    "o": 17,
    "p": 18,
    "q": 19,
    "r": 20,
    "s": 21,
    "t": 22,
    "u": 23,
    "v": 24,
    "w": 25,
    "x": 26,
    "y": 27,
    "z": 28,
    "æ": 29,
    "ø": 30,
    "å": 31,
    "é": 32,
    "ü": 33,
}


def numbers_to_words(reference):
    pass


def all_in_vocab(reference):
    has_all_in_vocab = True
    for c in reference:
        if c not in vocab:
            print(c)
            has_all_in_vocab = False
            break
    return has_all_in_vocab


def ffmpeg_extract_snippet_from_file(
    in_path: str, out_path: Path, start: int, duration: int, overwrite_file: str = "-n"
) -> None:
    """Call ffmpeg from command line to extract a snippet from an audio file

    Args:
        in_path (str): path of the file to extract from
        out_path (Path):  path to write the chunk to
        start (int): how many seconds in to the file to start the chunk.
        duration (int): how long the chunk should be (in seconds).
        overwrite_file (str): whether to overwrite already existing files. Can be either -y or -n
    """

    os.system(
        f"ffmpeg {overwrite_file} -hide_banner -loglevel error -ss {start} -i {in_path} -t {duration} {out_path}"
    )


def add_filename_col_and_extract_snippet(df: pd.DataFrame) -> pd.DataFrame:
    filename = f"{df['Pair']}_{df['Session']}"
    # df.name gets the index
    out_name = f"{filename}_{df.name}.wav"
    out_path = SAVE_DIR / out_name

    in_path = BASE_DIR / DATA_DIR / (filename + ".wav")
    ffmpeg_extract_snippet_from_file(
        in_path=in_path,
        out_path=out_path,
        start=df["starttime"],
        duration=df["Duration"],
    )
    df["filename"] = out_name
    return df


def mark_non_files(df: pd.DataFrame) -> pd.DataFrame:
    filename = f"{df['Pair']}_{df['Session']}"
    in_path = BASE_DIR / DATA_DIR / (filename + ".wav")
    df["file_exists"] = in_path.exists()
    return df


def sort_dict_by_key_length(d: dict):
    sorted_d = {}
    for k in sorted(d, key=len, reverse=True):
        sorted_d[k] = d[k]
    return sorted_d


def clean_reference(reference: pd.Series):

    reference = reference.str.replace("xx*", "")
    # remove things in between parentheses
    reference = reference.str.replace(r"\([^)]*\)", "")
    # remove things in betweten square brackets
    reference = reference.str.replace("[\(\[].*?[\)\]]", "")

    # remove sounds
    reference = reference.str.replace(r"host", "")

    reference = reference.str.replace(".", "")
    reference = reference.str.replace(",", "")
    reference = reference.str.replace("?", "")
    reference = reference.str.replace(":", "")
    reference = reference.str.replace(";", "")
    reference = reference.str.replace("!", "")
    reference = reference.str.replace("»", "")
    reference = reference.str.replace("«", "")
    reference = reference.str.replace("'", "")
    reference = reference.str.replace('"', "")
    reference = reference.str.replace("→", "")
    reference = reference.str.replace("↗", "")
    reference = reference.str.replace("í", "i")
    reference = reference.str.replace("&", "")
    reference = reference.str.replace(">", "")
    reference = reference.str.replace("<", "")
    reference = reference.str.replace("\uf0e0", "")
    reference = reference.str.replace(")", "")
    reference = reference.str.replace("“", "")
    reference = reference.str.replace("”", "")

    reference = reference.str.replace("[\r][\n]", " ")
    reference = reference.str.replace("/", " ")
    reference = reference.str.replace("-", " ")
    reference = reference.str.replace("—", " ")
    reference = reference.str.replace("–", " ")
    reference = reference.str.replace("ó", "o")
    reference = reference.str.replace("%", " procent")
    reference = reference.str.replace("à", "a")
    reference = reference.str.replace("ä", "æ")
    reference = reference.str.replace("ö", "æ")

    # Replace numbers with string
    numbers = reference.str.extractall("([\d]+)").dropna()
    unique_numbers = numbers[0].unique()
    num2word_dict = {n: num2words(n, lang="dk") for n in unique_numbers}
    # sort dict by number length for correct replacement of values
    num2word_dict = sort_dict_by_key_length(num2word_dict)
    for n, w in num2word_dict.items():
        reference = reference.str.replace(n, w)

    # Multiple spaces
    reference = reference.str.replace(" +", " ")

    reference = reference.str.lower()
    return reference


if __name__ == "__main__":

    BASE_DIR = Path("/work") / "data" / "speech-finetuning" / "puzzle_of_danish" / "puzzle-of-danish" / "christina-data"
    DATA_DIR = BASE_DIR / "PoD_sound"
    SAVE_DIR = Path("/work") / "data" / "speech-finetuning" / "puzzle_of_danish" / "puzzle-of-danish_segmented"
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir()
        
    df = pd.read_csv(BASE_DIR / "clean_data300821.txt", sep="\t")

    # A bit of cleaning
    df["Duration"] = df["Duration"].astype(str).str.rstrip("S").astype(float)
    df = df[
        (df["Duration"] > 3)
        & (df["Language"] == "Dansk")
        & ~(df["Pair"].isna())
        & ~(df["Session"].isna())
        & ~(df["Transcription"].isna())
    ]

    df["starttime"] = df["starttime"].astype(str).str.rstrip("S").astype(float)
    df["endtime"] = df["endtime"].astype(str).str.rstrip("S").astype(float)
    df["Pair"] = df["Pair"].astype(int).astype(str)
    df["Session"] = df["Session"].astype(int).astype(str)

    # drop bad rows
    df = df.drop([3675])
    # drop rows with no matching files
    df = df.apply(mark_non_files, axis=1)
    print(f"Num transcripts: {df.shape[0]}")
    print(f"Num transcripts not matching: {df.shape[0] - df['file_exists'].sum()}")

    df = df[df["file_exists"] == True]

    # clean transcripts
    df["Transcription"] = clean_reference(df["Transcription"])
    # Check if all characters in vocab
    has_all = df["Transcription"].apply(lambda x: all_in_vocab(x))

    transcriptions = df["Transcription"].tolist()
    for idx, has_a in enumerate(has_all):
        if not has_a:
            print(transcriptions[idx])

    # ~ 14.5 hours of speech with a duration longer than 3 seconds
    dur = df["Duration"].sum() / 60 / 60
    print(f"Total duration:  {dur}")

    # Segment audio files into utterances
    df = df.apply(add_filename_col_and_extract_snippet, axis=1)
    df = df[
        ["Pair", "Session", "Interlocutor", "Transcription", "Duration", "filename"]
    ]
    
    # add file duration
    pod["path"] = pod["filename"].apply(lambda x: SAVE_DIR / x)
    pod["file_duration"] = pod["path"].apply(lambda x: librosa.get_duration(filename=x))
    pod["time_dif"] = pod["Duration"] - pod["file_duration"]

    df.to_csv(BASE_DIR.parent.parent / "pod_data.csv")
