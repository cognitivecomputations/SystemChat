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
out_file = "user_intents_arabic.txt"

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


in_things = open(in_file, "r").readlines()
q = queue.Queue()
for item in in_things:
    q.put(item)

lock = threading.Lock()

# for thing in in_things:
#     prompt = f"### BEGIN PHRASE ###\n\n{thing}\n\n### END PHRASE ###\n\nPlease translate the preceding phrase from English to Arabic, in the style of Abu Dhabi / Dubai common dialect, casual yet professional tone.  To not provide any commentary - just a single line of text containing the translated phrase."
#     response = generate_openai_response(messages=[
#         {"role": "user", "content": prompt}
#     ])

#     arabic_version = response.choices[0].message.content.strip()
    
#     with open(out_file, "a", encoding="utf-8") as f:
#         f.write(arabic_version)
#         f.write("\n")

def do_translate():
    while True:
        thing = q.get()
        if thing is None: 
            break
        prompt = f"### BEGIN PHRASE ###\n\n{thing}\n\n### END PHRASE ###\n\nPlease translate the preceding phrase from English to Arabic, in the style of Abu Dhabi / Dubai common dialect, casual yet professional tone.  To not provide any commentary - just a single line of text containing the translated phrase."
        response = generate_openai_response(messages=[
            {"role": "user", "content": prompt}
        ])

        arabic_version = response.choices[0].message.content.strip()
        with lock:
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(arabic_version)
                f.write("\n")
            
        q.task_done()
    
        
with ThreadPoolExecutor(max_workers=100) as executor:
    for _ in range(100):
        executor.submit(do_translate)
    