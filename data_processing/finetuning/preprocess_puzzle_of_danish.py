import pandas as pd

from pathlib import Path
import os


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
    ffmpeg_extract_snippet_from_file(
        in_path=f"{filename}.wav",
        out_path=out_path,
        start=df["starttime"],
        duration=df["Duration"],
    )
    df["filename"] = out_name
    return df


def clean_transcript(transcripts: pd.Series):
    transcripts = transcripts.str.lower()
    transcripts = transcripts.str.replace("xx*", "")
    # remove things in between parentheses
    transcripts = transcripts.str.replace(r"\([^)]*\)", "")
    # remove anything that is not a character
    transcripts = transcripts.str.replace("[^\w\s]", "")
    # remove double whitespace
    transcripts = transcripts.str.replace(r"\s+", " ")
    return transcripts


if __name__ == "__main__":

    BASE_DIR = Path("/data") / "puzzle-of-danish" / "christina-data"
    DATA_DIR = BASE_DIR / "PoD_sound"
    SAVE_DIR = BASE_DIR / "segmented_audio"
    if not SAVE_DIR.exists():
        # SAVE_DIR.mkdir()
        pass

    pd.set_option("display.max_columns", None)

    df = pd.read_csv(BASE_DIR / "clean_data300821.txt", sep="\t")

    # A bit of cleaning
    df["Duration"] = df["Duration"].astype(str).str.rstrip("S").astype(float)
    df = df[
        (df["Duration"] > 3)
        & (df["Language"] == "Dansk")
        & ~(df["Pair"].isna())
        & ~(df["Session"].isna())
    ]

    df["starttime"] = df["starttime"].astype(str).str.rstrip("S").astype(float)
    df["endtime"] = df["endtime"].astype(str).str.rstrip("S").astype(float)
    df["Pair"] = df["Pair"].astype(int).astype(str)
    df["Session"] = df["Session"].astype(int).astype(str)

    # clean transcripts a bit
    df["Transcription"] = clean_transcript(df["Transcription"])
    # remove double whitespace

    # ~ 16.5 hours of speech with a duration longer than 3 seconds
    df["Duration"].sum() / 60 / 60

    df = df.apply(add_filename_col_and_extract_snippet, axis=1)
    df = df[
        ["Pair", "Session", "Interlocutor", "Transcription", "Duration", "filename"]
    ]

    df.to_csv(BASE_DIR / "clean_data_transcript_filenames.csv")
