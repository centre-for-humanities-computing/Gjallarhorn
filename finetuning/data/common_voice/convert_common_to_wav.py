import librosa
from datasets import load_dataset
import os
import soundfile as sf

dataset = load_dataset("mozilla-foundation/common_voice_9_0", "da", cache_dir="./", use_auth_token=True)

def contains_number(reference):
    return re.search("[0-9]+", reference)

def clean_reference(reference):
    reference = reference.replace(".", "")
    reference = reference.replace(",", "")
    reference = reference.replace("?", "")
    reference = reference.replace(":", "")
    reference = reference.replace(";", "")
    reference = reference.replace("!", "")
    reference = reference.replace("’", "")
    reference = reference.replace("'", "")
    reference = reference.replace(" -", "")
    reference = reference.replace(" –", "")
    reference = reference.replace("\"", "")
    reference = reference.replace("/", " ")
    reference = reference.replace(".", "")
    reference = reference.replace("”", "")
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
        tsv.append(f"{audio_file_out}\t{len(resampled_audio)}")

        sentence = clean_reference(sentence)

        wrd.append(sentence)
        
        ltr_entry = " ".join(sentence.replace(" ", "|"))
        ltr.append(ltr_entry)
        
        if os.path.exists(audio_file_out):
            print("BAD FILE HANDLE")
        
        sf.write(audio_file_out, resampled_audio, samplerate=16000) 
        if i != 0 and i % 50 == 0:
            with open(f"./manifest/{data_handle}.tsv", "w", encoding="utf-8") as f:
                f.write("\n".join(tsv))
            
            with open(f"./manifest/{data_handle}.wrd", "w", encoding="utf-8") as f:
                f.write("\n".join(wrd))

            with open(f"./manifest/{data_handle}.ltr", "w", encoding="utf-8") as f:
                f.write("\n".join(ltr))

            break
        
        


download_data_split("test") 
