import os
import sys
from subprocess import Popen

jobs = []

all_dirs = os.listdir("/work/data/p1-r24syv/files/24syv")
processes_output_dir = "./processes_output"

for dir in all_dirs:
    if dir == "2011-11":
        continue

    out_dir = f"/work/data/p1-r24syv-segmented/r24syv/{dir}"

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    in_dir = f"/work/data/p1-r24syv/files/24syv/{dir}"
    log_file = f"{processes_output_dir}/{dir}.logfile"

    # Create file
    with open(log_file, "w", encoding="utf-8") as f:
        pass

    process = Popen(
        [str(sys.executable)] + ['process_dir.py'] + ["--directory_in"] + [in_dir] + ["--directory_out"] + [out_dir],
        stdout=log_file
    )

    jobs.append(process)

print("ALl processes executed.")
for p in jobs:
    p.wait()
