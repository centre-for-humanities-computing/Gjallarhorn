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
    process = Popen(
        [str(sys.executable)] + ['process_dir.py'] + ["--directory_in"] + [in_dir] + ["--directory_out"] + [out_dir],
        stdout=f"{processes_output_dir}/{dir}.logfile"
    )

    jobs.append(process)

print("ALl processes executed.")
for p in jobs:
    p.wait()
