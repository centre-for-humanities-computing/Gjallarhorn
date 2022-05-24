import os
import sys
from subprocess import Popen

jobs = []
file_outs = []

all_dirs = os.listdir("/work/data/p1-r24syv/files/drp1")
processes_output_dir = "./processes_output_p1"

for dir in all_dirs:
    out_dir = f"/work/data/p1-r24syv-segmented/p1/{dir}"

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    in_dir = f"/work/data/p1-r24syv/files/drp1/{dir}"
    log_file = f"{processes_output_dir}/{dir}.logfile"

    # Create file
    file_out = open(log_file, "w", encoding="utf-8")

    process = Popen(
        [str(sys.executable)] + ['process_dir.py'] + ["--directory_in"] + [in_dir] + ["--directory_out"] + [out_dir],
        stdout=file_out
    )

    file_outs.append(file_out)
    jobs.append(process)

print("ALl processes executed.")
for p in jobs:
    p.wait()

for file_out in file_outs:
    file_out.close()
