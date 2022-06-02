import os


filename_p1 = "/work/data/p1-r24syv-dedup/metadata/p1_no_reruns.txt"
filename_24syv = "/work/data/p1-r24syv-dedup/metadata/r24syv_no_reruns.txt"

new_dir = "/work/data/p1-r24syv-segmented/p1/"

with open(filename_p1, "r", encoding="utf-8") as f:
    all_lines = f.read().split("\n")

counter_all = 0
counter_processed = 0
counter_not_yet = 0

for i, line in enumerate(all_lines):
    try:
        splitted_line = line.split("/")
        filename_mp3 = splitted_line[-1]
        if ".mp3" in filename_mp3:
            filename_mp3 = filename_mp3[:-4]
    
        internal_dir = splitted_line[-2]
        processed_path = f"{new_dir}{internal_dir}/{filename_mp3}"
        if os.path.exists(processed_path) and os.path.exists(processed_path + "/metadata.tsv"):
            counter_processed += 1
        elif os.path.exists(processed_path[:-4]) and os.path.exists(processed_path[:-4] + "/metadata.tsv"):
            counter_processed += 1
        else:
            counter_not_yet += 1
        counter_all += 1
    except Exception as e:
        print(e)

    
    if i % 100 == 0:
        print(counter_processed)
        print(counter_not_yet)
        print(f"processed: {i}/{len(all_lines)}")


print(counter_all)
print(counter_processed)
print(counter_not_yet)

