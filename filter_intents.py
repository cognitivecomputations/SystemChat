import jsonlines
import json
from tqdm import tqdm
import json
import random
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from tenacity import retry, wait_exponential, stop_after_attempt
import queue

in_file = "user_intents.txt"
out_file = "user_intents_filtered.txt"

load_dotenv()
openai_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("DEPLOYMENT_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL_NAME")
azure_endpoint = os.getenv("AZURE_ENDPOINT")

client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=openai_api_version,
    azure_endpoint=azure_endpoint
)

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10))
def generate_openai_response(messages, max_tokens=2000):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response

intents = open(in_file, "r").readlines()
q = queue.Queue()
for intent in intents:
    q.put(intent)

lock = threading.Lock()

def do_check_intents():
    while True:
        intent = q.get().strip()
        print("intent", intent)
        if intent is None: 
            break
        prompt = f"Regarding the following usecase, please determine whether it can be answered by a text-based chatbot (like ChatGPT) with no functions or external integrations.  Reply true or false, with no other commentary.\nThe usecase is: {intent}"
        response = generate_openai_response([{"role": "user", "content":prompt}], 5)
        usable = "true" in response.choices[0].message.content.strip().lower()
        if usable:
            with lock:
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(intent)
                    f.write("\n")
                    f.flush()
                    f.close()
        q.task_done()
            
def main():
    # do_check_intents()
    with ThreadPoolExecutor(max_workers=1000) as executor:
        for _ in range(1000):
            executor.submit(do_check_intents)

if __name__ == "__main__":
    main()