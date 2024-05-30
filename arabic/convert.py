import jsonlines
import json
from tqdm import tqdm

in_file = "SystemChatArabic.jsonl"
out_file = "SystemChatArabic2.jsonl"

with open(out_file, "w", encoding="utf-8") as f:
    with jsonlines.open(in_file) as reader:
        for obj in tqdm(reader):
            json.dump({"messages":obj}, f)
            f.write("\n")