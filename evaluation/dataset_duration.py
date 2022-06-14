import pandas as pd
from datasets import load_dataset
from data.load_puzzle_of_danish import load_puzzle_of_danish

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

def get_duration(example):
    dur = example["audio"]["array"].shape[0] / example["audio"]["sampling_rate"]
    example["duration"] = dur
    return example
cv9 = cv9.map(get_duration, num_proc=6)


## PoD
pod = load_puzzle_of_danish(return_df=True)
pod["id"] = pod["Pair"].astype(str) + pod["Interlocutor"]

n_speakers = pod["id"].nunique()
n_sentences = pod.shape[0]
n_hours = (pod["Duration"].sum() / 60 / 60).round(2)

pod_data = pd.DataFrame(
    {
        "Dataset" : ["Puzzle of Danish"],
        "Split" : ["All"],
        "Sentences" : [n_sentences],
        "Speakers" : [n_speakers],
        "Hours" : [n_hours]
    }
)

cv