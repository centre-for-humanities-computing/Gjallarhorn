from datasets import load_dataset
import re
from wasabi import msg

def clean_reference(example):
    reference = example["sentence"]
    reference = reference.replace(".", "")
    reference = reference.replace(",", "")
    reference = reference.replace("?", "")
    reference = reference.replace(":", "")
    reference = reference.replace(";", "")
    reference = reference.replace("!", "")
    reference = reference.replace("»", "")
    reference = reference.replace("«", "")
    reference = reference.replace("\'", "")
    reference = reference.replace("\"", "")
    reference = reference.replace("í", "i")

    # TODO: Discuss this with lasse
    reference = reference.replace("-", " ")
    reference = reference.replace("—", " ")
    reference = reference.replace("–", " ")
    reference = reference.replace("ó", "o")
    # Multiple spaces
    reference = re.sub(" +", " ", reference)

    reference = reference.lower()
    example["sentence"] = reference
    return example


def load_common_voice():
    msg.info("Loading Common Voice..")
    cv = load_dataset("mozilla-foundation/common_voice_9_0", "da", split="test", use_auth_token=True)
    cv = cv.map(lambda example: clean_reference(example), num_proc=8)
    msg.good("Common Voice loaded!")
    return cv["path"], cv["sentence"]
