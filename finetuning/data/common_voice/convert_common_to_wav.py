import os
import re

import librosa
import soundfile as sf
from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_9_0", "da", cache_dir="./", use_auth_token=True)

vocab = {"<pad>": 0, "<unk>": 1, " ": 2, "a": 3, "b": 4, "c": 5, "d": 6, "e": 7, "f": 8, "g": 9, "h": 10, "i": 11,
         "j": 12, "k": 13, "l": 14, "m": 15, "n": 16, "o": 17, "p": 18, "q": 19, "r": 20, "s": 21, "t": 22, "u": 23,
         "v": 24, "w": 25, "x": 26, "y": 27, "z": 28, "æ": 29, "ø": 30, "å": 31, "é": 32, "ü": 33}

print(dataset)


def all_in_vocab(reference):
    has_all_in_vocab = True
    for c in reference:
        if c not in vocab:
            print(c)
            has_all_in_vocab = False
            break
    return has_all_in_vocab


def contains_number(reference):
    return re.search("[0-9]+", reference)


def clean_reference(reference):
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
    return reference


def download_data_split(data_handle):
    ltr = []
    tsv = ["dummy_path"]
    wrd = []

    for i, item in enumerate(dataset[data_handle]):
        audio = item["audio"]["array"]
        resampled_audio = librosa.resample(audio, orig_sr=item["audio"]["sampling_rate"], target_sr=16000)
        audio_id = item["path"].split("/")[-1][0:-4] + ".wav"
        sentence = item["sentence"]
        audio_file_out = f"{data_handle}/{audio_id}"
        tsv.append(f"common_voice/{audio_file_out}\t{len(resampled_audio)}")

        sentence = clean_reference(sentence)
        has_all = all_in_vocab(sentence)
        if not has_all:
            print(sentence)
            print(audio_file_out)

        wrd.append(sentence)

        ltr_entry = " ".join(sentence.replace(" ", "|"))
        ltr.append(ltr_entry)

        if os.path.exists(audio_file_out):
            print("BAD FILE HANDLE")

        sf.write(audio_file_out, resampled_audio, samplerate=16000)
        if i != 0 and i % 50 == 0:
            print(f"Progress: {i}/{len(dataset[data_handle])}")
            with open(f"./manifest/{data_handle}.tsv", "w", encoding="utf-8") as f:
                f.write("\n".join(tsv))

            with open(f"./manifest/{data_handle}.wrd", "w", encoding="utf-8") as f:
                f.write("\n".join(wrd))

            with open(f"./manifest/{data_handle}.ltr", "w", encoding="utf-8") as f:
                f.write("\n".join(ltr))


handles = ['train', 'test', 'other', 'validation']

for handle in handles:
    os.makedirs(handle)
    download_data_split(handle)
