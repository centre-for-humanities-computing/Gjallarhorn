from collections import defaultdict
from pathlib import Path
from typing import List

import librosa
import pandas as pd
from pandas.io.formats.style import Styler
import soundfile as sf
from datasets import load_dataset

from data.load_common_voice import load_common_voice
from data.load_nst_test import load_nst_data
from data.load_puzzle_of_danish import load_puzzle_of_danish
from huggingsound import SpeechRecognitionModel

from wasabi import Printer

def calc_performance(model, references: List[str], predictions: List[dict]):
    references = [
        {"transcription": t} for t in references
    ]
    wer = model.evaluate(references=references, predictions=predictions)
    return wer


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    model_ids = ["Alvenir/wav2vec2-base-da-ft-nst"]

    nst_files, nst_references = load_nst_data()
    pod_files, pod_references = load_puzzle_of_danish()
    cv_files, cv_references = load_common_voice()

    data_paths = [nst_files, pod_files, cv_files]
    data_references = [nst_references, pod_references, cv_references]
    data_sets = ["NST", "PoD", "CV"]

    performance = defaultdict(lambda: {})
    for model_id in model_ids:
        msg.divider(f"Evaluating {model_id}")
        model = SpeechRecognitionModel(model_id, device="cuda")
        for files, references, data_set in zip(data_paths, data_references, data_sets):
            
            with msg.loading(f"Transcribing {data_set} with {model_id}..."):
                transcriptions = model.transcribe(files, batch_size=25)
            msg.good(f"Finished transcribing {data_set} with {model_id}!")
            
            with msg.loading(f"Calculating wer and cer..."):
                perf = calc_performance(model, references, transcriptions)
            msg.good("Finished calculating wer and cer!")
            performance[data_set][model_id] = perf

    df = pd.DataFrame.from_dict({(i,j): performance[i][j] 
                           for i in performance.keys() 
                           for j in performance[i].keys()},
                       orient='index')

    df = df.rename_axis(["dataset", "model"]).reset_index()
    df.to_csv("transcription_performance.csv")

    df_p = df.pivot(index="model", columns="dataset", values=["wer", "cer"])
    df_p = df_p.rename_axis("")


    print(df_p.to_latex(multicolumn=True, float_format="%.2f"))

    s = Styler(df_p, precision=2).highlight_max(axis=0, props='bfseries: ;').to_latex(hrules=True, multicol_align="c")
    print(s)

    # print(df_p.to_latex(hrules=True, sparse_index=True))

    # s = df_p.style.highlight_max(axis=0, props='cellcolor:{red}; bfseries: ;')
    # print(s.to_latex())

    