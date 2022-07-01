
from pathlib import Path
import pandas as pd

import librosa
from wasabi import msg

POD_PATH = Path("/work") / "data" / "speech-finetuning" / "puzzle_of_danish"
SAVE_DIR = Path("/work/data/speech-finetuning") / "puzzle_of_danish" / "puzzle-of-danish_segmented"


def load_puzzle_of_danish(return_df=False):
    msg.info("Loading Puzzle of Danish...")
    pod = pd.read_csv(POD_PATH / "pod_data.csv")
    # dropping 10 very long audio segments to avoid oom erros
    pod = pod[pod["Duration"] < 60]
    # dropping very short files
    pod = pod[pod["file_duration"] >= 2]
    # dropping nas
    pod = pod.dropna(subset=["Transcription"])

    msg.good("Puzzle of Danish loaded!")
    if return_df:
        return pod
    return pod["path"].tolist(), pod["Transcription"].tolist()




if __name__ == "__main__":
    pod = pd.read_csv(POD_PATH / "pod_data.csv")

    pod["path"] = pod["filename"].apply(lambda x: SAVE_DIR / x)
    pod["file_duration"] = pod["path"].apply(lambda x: librosa.get_duration(filename=x))
    pod["time_dif"] = pod["Duration"] - pod["file_duration"]

    pod.to_csv(POD_PATH.parent.parent / "pod_data.csv")



