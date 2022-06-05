import soundfile as sf
from datasets import load_dataset
import librosa

from huggingsound import SpeechRecognitionModel
import pandas as pd

from pathlib import Path


model_id = "Alvenir/wav2vec2-base-da-ft-nst"
SAVE_DIR = Path("/data") / "puzzle_of_danish_segmented"


# nst = load_dataset("Alvenir/nst-da-16khz", split="test")

pod = pd.read_csv("clean_data_transcript_filenames.csv")
# dropping 10 very long audio segments to avoid oom erros
pod = pod[pod["Duration"] < 60]
pod["path"] = pod["filename"].apply(lambda x: SAVE_DIR / x)
pod["file_duration"] = pod["path"].apply(lambda x: librosa.get_duration(filename=x))
pod["time_dif"] = pod["Duration"] - pod["file_duration"]


print(pod.shape[0])
pod = pod[pod["file_duration"] >= 3]
print(pod.shape[0])

paths = [SAVE_DIR / p for p in pod["filename"].tolist()]

model = SpeechRecognitionModel(model_id, device="cuda")
# transcriptions = model.transcribe(paths[:5])

# print([t["transcription"] for t in transcriptions])

eval_dict = [
    {"transcription": t, "path": p}
    for t, p in zip(pod["Transcription"].tolist(), paths)
]
wer = model.evaluate(eval_dict)
print(wer)
