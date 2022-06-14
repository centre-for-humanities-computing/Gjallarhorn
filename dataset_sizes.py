import pandas as pd
from datasets import load_dataset


pd.set_option("display.max_columns", None)

nst = pd.read_csv("NST_dk.csv")
sup = pd.read_csv("supplement_dk.csv")


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
cv9 = cv9.map(get_duration, num_proc=8)
