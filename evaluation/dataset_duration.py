import pandas as pd
from datasets import load_dataset
from data.load_puzzle_of_danish import load_puzzle_of_danish
from librosa import get_duration

from pandas.io.formats.style import Styler



def get_file_duration(path):
    dur =  get_duration(filename=path)
    
    return {"dur" : dur}


def get_hour_duration(dur_dataset):
    dur = sum(dur_dataset["dur"]) / 60 / 60
    dur = round(dur, 2)
    return dur

# Thanks to Danspeech thesis <3 
nst_data = pd.DataFrame(
    {
        "Dataset": ["NST"] * 3,
        "Split": ["Train", "Validation", "Test"],
        "Sentences": [312, 312, 987],
        "Speakers": [465, 115, 56],
        "Hours": [192.2, 48.7, 76.8],
    }
)


cv9 = load_dataset(
    "mozilla-foundation/common_voice_9_0", "da", use_auth_token=True
)

cv_train = cv9["train"]
cv_test = cv9["test"]

cv_train_n_sentences = len(cv_train)
cv_test_n_sentences = len(cv_test)

cv_train_dur = cv_train.map(get_file_duration, input_columns="path", num_proc=20, remove_columns=cv_train.column_names)
cv_test_dur = cv_test.map(get_file_duration, input_columns="path", num_proc=20, remove_columns=cv_test.column_names)


cv9_data = pd.DataFrame({
    "Dataset" : ["Common Voice 9"] * 2,
    "Split" : ["Train", "Test"],
    "Sentences" : [cv_train_n_sentences, cv_test_n_sentences],
    "Speakers" : ["NA", "NA"],
    "Hours" : [get_hour_duration(cv_train_dur), get_hour_duration(cv_test_dur)]
})


## PoD
pod = load_puzzle_of_danish(return_df=True)
pod["id"] = pod["Pair"].astype(str) + pod["Interlocutor"]

n_speakers = pod["id"].nunique()
n_sentences = pod.shape[0]
n_hours = (pod["Duration"].sum() / 60 / 60).round(2)

pod_data = pd.DataFrame(
    {
        "Dataset" : ["Puzzle of Danish"],
        "Split" : ["Test"],
        "Sentences" : [n_sentences],
        "Speakers" : [n_speakers],
        "Hours" : [n_hours]
    }
)

## alvenir
alvenir = load_dataset("Alvenir/alvenir_asr_da_eval")
alvenir = alvenir["test"]
alv_n_sentences = len(alvenir)


alv_seconds = alvenir.map(get_file_duration, input_columns="path", num_proc=20, remove_columns=alvenir.column_names)


alvenir_data = pd.DataFrame(
    {
        "Dataset": ["Alvenir"],
        "Split": ["Test"],
        "Sentences": [alv_n_sentences],
        "Speakers": [50],
        "Hours": [get_hour_duration(alv_seconds)],
    }
)

joined = pd.concat([nst_data, cv9_data, pod_data, alvenir_data], axis=0).reset_index(drop=True)

print(joined.to_latex())

s = (
        Styler(joined, precision=2)
        .to_latex(hrules=True)
    )
print(s)