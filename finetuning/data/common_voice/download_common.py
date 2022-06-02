
from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_9_0", "da", cache_dir="./", use_auth_token=True)

