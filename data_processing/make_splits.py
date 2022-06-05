import random

final_lines = []

with open("/work/data/p1-r24syv-segmented/metadata_final.tsv", "r", encoding="utf-8") as f:
    all_lines = f.readlines()
    for i, line in enumerate(all_lines):
        if "///" in line or "manifest" in line:
            continue
       
        if line[0:2] == "//":
            line = line[2:]
        
        line = line.rstrip("\n")
        
        if line[0] == "/":
            line = line[1:]
            # print(line)
        
        if len(line.split("/")) != 4:
            print("length of line failed")
            print(line)
            continue
        
        if len(line.split("\t")) != 2:
            print("tab fail")
            print(line)
            continue

        final_lines.append(line)

        if i % 10000 == 0:
            print(f"{i}/{len(all_lines)}")

random.shuffle(final_lines)

validation = final_lines[0:8000]
train = final_lines[8000:]

with open("/work/data/p1-r24syv-segmented/manifest/train.tsv", "w", encoding="utf-8") as f:
    f.write("/work/data/p1-r24syv-segmented/\n")
    f.write("\n".join(train))

with open("/work/data/p1-r24syv-segmented/manifest/valid.tsv", "w", encoding="utf-8") as f:
    f.write("/work/data/p1-r24syv-segmented/\n")
    f.write("\n".join(validation))

