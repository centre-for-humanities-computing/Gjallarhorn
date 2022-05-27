import os
import sys
from subprocess import Popen

jobs = []
file_outs = []

all_dirs = os.listdir("/work/data/p1-r24syv/files/drp1")
processes_output_dir = "./processes_output_p1"

for dir in all_dirs:
    if not dir in [
        "2005-12", "2006-01", "2007-01", "2007-10",
        "2007-10", "2008-05", "2008-06", "2007-08", "2008-10",
        "2008-10", "2009-02", "2009-03", "2009-08", "2010-02",
        "2010-09", "2011-04", "2011-11", "2012-03", "2012-09",
        "2012-09", "2014-08", "2015-04", "2012-06", "2012-07",
        "2016-02", "2016-06", "2017-07", "2018-02", "2018-05",
        "2018-07", "2018-08", "2019-01", "2019-03", "2019-08",
        "2019-09", "2020-05", "2020-11", "2021-02", "2021-09",
        "2021-11", "2021-12"
    ]:
        continue

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
