from datasets import load_dataset
from wasabi import msg


def load_alvenir_eval():
    msg.info("Loading Alvenir ASR eval..")
    alvenir = load_dataset("Alvenir/alvenir_asr_da_eval")
    msg.good("Common Voice loaded!")
    return alvenir["audio"]["path"], alvenir["sentence"]
