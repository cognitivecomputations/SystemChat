import argparse
import jsonlines
import json
from tqdm import tqdm
import uuid
import random

in_file = "SystemChat_multilingual.jsonl"
out_file = "SystemChat_multilingual_filtered.jsonl"

with open(out_file, "w", encoding="utf-8") as f:
    with jsonlines.open(in_file) as reader:
        for obj in tqdm(reader):
            if obj[-1]["role"] == "user":
                obj.pop()
            new_obj = {"messages":obj}
            if new_obj is None:
                continue
            json.dump(new_obj, f)
            f.write("\n")


