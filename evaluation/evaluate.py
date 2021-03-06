from collections import defaultdict
from pathlib import Path
from typing import List

import librosa
import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset
from huggingsound import SpeechRecognitionModel
from pandas.io.formats.style import Styler
from wasabi import Printer

from data.load_common_voice import load_common_voice
from data.load_nst_test import load_nst_data
from data.load_puzzle_of_danish import load_puzzle_of_danish
from data.load_alvenir_eval import load_alvenir_eval

from huggingsound import SpeechRecognitionModel



def calc_performance(model, references: List[str], predictions: List[dict]):
    references = [{"transcription": t} for t in references]
    wer = model.evaluate(references=references, predictions=predictions)
    return wer


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    model_ids = ["chcaa/xls-r-300m-danish-nst-cv9", "chcaa/xls-r-300m-nst-cv9-da", "chcaa/alvenir-wav2vec2-base-da-nst-cv9", "Alvenir/wav2vec2-base-da-ft-nst"]

    nst_files, nst_references = load_nst_data()
    pod_files, pod_references = load_puzzle_of_danish()
    cv_files, cv_references = load_common_voice()
    alv_files, alv_references = load_alvenir_eval()

    data_paths = [alv_files, nst_files, pod_files, cv_files]
    data_references = [alv_references, nst_references, pod_references, cv_references]
    data_sets = ["Alvenir", "NST", "PoD", "CV"]

    performance = defaultdict(lambda: {})
    for model_id in model_ids:
        msg.divider(f"Evaluating {model_id}")
        model = SpeechRecognitionModel(model_id, device=torch.device("cuda"), use_auth_token=True)
        for files, references, data_set in zip(data_paths, data_references, data_sets):
            with msg.loading(f"Transcribing {data_set} with {model_id}..."):
                transcriptions = model.transcribe(files, batch_size=10)
            msg.good(f"Finished transcribing {data_set} with {model_id}!")

            with msg.loading(f"Calculating wer and cer..."):
                perf = calc_performance(model, references, transcriptions)
            msg.good("Finished calculating wer and cer!")
            msg.text(perf)
            performance[data_set][model_id] = perf

    df = pd.DataFrame.from_dict(
        {
            (i, j): performance[i][j]
            for i in performance.keys()
            for j in performance[i].keys()
        },
        orient="index",
    )

    df = df.rename_axis(["dataset", "model"]).reset_index()
    df.to_csv("transcription_performance_alvenir.csv")
