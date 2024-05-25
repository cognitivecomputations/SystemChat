import argparse
import jsonlines
import json
from tqdm import tqdm
import uuid
import random

in_file = "SystemChat_copy_filtered.jsonl"
out_file = "SystemChat_sharegpt.jsonl"

def convert_sample(sample, id = None):
    
    obj = {
        "conversations": []
    }
    
    for turn in sample:
        if turn["role"] == "system":
            obj["conversations"].append({"from": "system", "value": turn["content"] })
        elif turn["role"] == "user":
            obj["conversations"].append({"from": "human", "value": turn["content"] })
        else:
            obj["conversations"].append({"from": "gpt", "value": turn["content"]})
        if turn["content"].strip() == "" or "<<||END||>>" in turn["content"]:
            return None
    return obj

with open(out_file, "w", encoding="utf-8") as f:
    with jsonlines.open(in_file) as reader:
        for obj in tqdm(reader):
            if obj[-1]["role"] == "user":
                obj.pop()
            new_obj = convert_sample(obj)
            if new_obj is None:
                continue
            json.dump(new_obj, f)
            f.write("\n")


