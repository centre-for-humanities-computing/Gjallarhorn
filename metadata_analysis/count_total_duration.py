"""Get the size of our dataset with reruns removed"""
import librosa
from pathlib import Path

from typing import Union, List
from multiprocessing import Pool


def get_file_duration(path: Union[str, Path]) -> float:
    """Get the duration of a file in hours

    Arguments:
        path {Union[str, Path]} -- Path to file

    Returns:
        float -- Duration in minutes
    """
    try:
        dur = librosa.get_duration(filename=path) / (60 * 2)
    except:
        print(f"error in file {path}")
        dur = 0
    return dur


def load_no_reruns(path: Union[str, Path]) -> List[str]:
    """Loads filepaths to data with reruns removed

    Arguments:
        path {Union[str, Path]} -- Path to data

    Returns:
        List[str] -- List of filepaths
    """
    with open(path, "r") as f:
        return f.read().splitlines()


if __name__ == "__main__":

    #### No reruns
    METADATA_PATH = Path("/work") / "data" / "p1-r24syv-dedup" / "metadata"
    SAVE_PATH = Path("/work") / "49978" / "Gjallarhorn" / "metadata_analysis" / "results"
    
    ALL_FILE_PATHS = Path("/work") / "data" / "p1-r24syv" / "files"
    
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir()

    p1_no_reruns = METADATA_PATH / "p1_no_reruns.txt"
    r24syv_no_reruns = METADATA_PATH / "r24syv_no_reruns.txt"

    p1_files = load_no_reruns(p1_no_reruns)
    r24syv_files = load_no_reruns(r24syv_no_reruns)
    voxpopuli = Path("/work") / "data" / "voxpopuli" / "unlabelled_data" / "da" 
    voxpopuli_filepaths = voxpopuli.rglob("*.ogg")

    pool = Pool()

    p1_hours = pool.map(get_file_duration, p1_files)
    print(f"Hours of data for P1: {sum(p1_hours)}")
    r24syv_hours = pool.map(get_file_duration, r24syv_files)
    print(f"Hours of data for Radio24syv: {sum(r24syv_hours)}")
    
    voxpopuli_hours = pool.map(get_file_duration, voxpopuli_filepaths)
    print(f"Hours of data for Voxpopuli: {sum(voxpopuli_hours)}")

    header = "channel,total_duration,no_reruns\n"
    with open(SAVE_PATH / "durations.csv", "w") as f:
        f.write(header)
        for dur, channel in zip([p1_hours, r24syv_hours, voxpopuli_hours], ["P1", "Radio24syv", "Voxpopuli"]):
            f.write(f"{channel},{sum(dur)},1\n")
    

    #### All data

    p1_all_files = ALL_FILE_PATHS / "drp1"
    p1_all_files = [x for x in p1_all_files.rglob("*") if x.is_file()]

    r24syv_all_files = ALL_FILE_PATHS / "24syv"
    r24syv_all_files = [x for x in r24syv_all_files.rglob("*") if x.is_file()]

    p1_all_files_hours = pool.map(get_file_duration, p1_all_files)
    r24syv_all_files_hours = pool.map(get_file_duration, r24syv_all_files)

    with open(SAVE_PATH / "durations.csv", "a") as f:
        for dur, channel in zip([p1_all_files_hours, r24syv_all_files_hours], ["P1", "Radio24syv"]):
            f.write(f"{channel},{sum(dur)},0\n")


