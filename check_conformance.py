import argparse
import jsonlines
import json
from tqdm import tqdm
import uuid
import random
import ollama

import ollama

in_file = "SystemChatMistral copy.jsonl"
out_file = "SystemChatMistral_checked.jsonl"

def convert_sample(sample, id = None):
    
    obj = {
        "conversations": []
    }

    directive = None
    grades = []
    
    for turn in sample:
        if turn["role"] == "system":
            obj["conversations"].append({"from": "system", "value": turn["content"] })
            directive = turn["content"]
        elif turn["role"] == "user":
            obj["conversations"].append({"from": "human", "value": turn["content"] })
        else:
            response = turn["content"]

            prompt = """Given the directive: <directive>{directive}</directive> and the response <response>{response}</response> - does the response conform to the directive? 
respond with a number from 1 to 5.  1 means the response completely ignores the directive.  5 means it completely obeys the objective. 
your response should be json like this: {"conformance":<score>,"reason":"<reason>"}"""

            grade = ollama.chat(model='llama3', format="json", messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            grade = json.loads(grade["message"]["content"])
            grades.append(grade["conformance"])       

            obj["conversations"].append({"from": "gpt", "value": turn["content"]})
        if turn["content"].strip() == "" or "<<||END||>>" in turn["content"]:
            return None
    obj["grades"] = grades
    print(json.dumps(obj), "\n")
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
