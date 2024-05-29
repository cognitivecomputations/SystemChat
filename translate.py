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

in_file = "languages.txt"
out_file = "multilingual_templates.jsonl"

load_dotenv()
openai_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("DEPLOYMENT_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL_NAME")
azure_endpoint = os.getenv("AZURE_ENDPOINT")

metaprompt = "Sam is a user of an AI chatbot.\nSam wants to accomplish the following abstract goal: '{usecase}'.\nThe following is the conversation history.\n<conversation_history>\n{conversation_history}\n</conversation_history>\nIf the conversation has fewer than 8 turns, Sam should come up with some further prompt.\nIf Sam would give up in frustration, or if his goal is satisfied by the conversation, respond <<||END||>> and nothing else.\nOtherwise, please assume role of Sam, and generate his next prompt to the AI assistant.  Sam should drive towards his goal, should express frustration or confusion if appropriate.\nYou should reply with just Sam's new prompt, in the first person, with no explanation or commentary.  (or <<||END||>> if Sam has nothing else to say)"
initialprompt = "Sam is a user of an AI chatbot.\nSam wants to accomplish the following abstract task: '{usecase}'.\nInvent an instance of this abstract goal, and then take the role of Sam and write a prompt from his perspective that he would use to try to accomplish his goal.  Respond with just Sam's prompt, no explanation.  Sam does not greet the AI assistant, nor say the word 'Hey'."


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


languages = open(in_file, "r").readlines()
q = queue.Queue()
for language in languages:
    q.put(language)

lock = threading.Lock()

def do_translate():
    while True:
        language = q.get()
        if language is None: 
            break
        
        response = generate_openai_response(messages=[
            {"role": "system", "content": "You are a translator. But never translate the words '{usecase}', '{conversation_history}', or '<<||END||>>' - those special words need to remain unchanged."},
            {"role": "user", "content": f"Please translate from English to {language}:\n\n {metaprompt}"}
        ])
        localized_metaprompt = response.choices[0].message.content.strip()
        
        response = generate_openai_response(messages=[
            {"role": "system", "content": "You are a translator. But never translate the words '{usecase}', '{conversation_history}', or '<<||END||>>' - those special words need to remain unchanged."},
            {"role": "user", "content": f"Please translate from English to {language}:\n\n {initialprompt}"}
        ])
        localized_initialprompt = response.choices[0].message.content.strip()
    
        with lock:
            with open(out_file, "a", encoding="utf-8") as f:
                obj = {"language":language.strip(), "metaprompt": localized_metaprompt, "initialprompt": localized_initialprompt}
                print(obj)
                json.dump(obj, f)
                f.write("\n")
                f.flush()
                f.close()
            
        q.task_done()
    
        
def main():
    # do_translate()
    with ThreadPoolExecutor(max_workers=1000) as executor:
        for _ in range(1000):
            executor.submit(do_translate)

if __name__ == "__main__":
    main()