import pandas as pd
from pathlib import Path


SAVE_PATH = Path("/work") / "49978" / "Gjallarhorn" / "metadata_analysis" / "results"


df = pd.read_csv(SAVE_PATH / "durations.csv")

# I fucked up the code. Converting back to seconds by multiplying with 120
# then to minutes, then to hours

df["total_duration"] = ((df["total_duration"] * 120) / 60 / 60).round().astype(int)


print(df.sort_values("channel").to_latex())
