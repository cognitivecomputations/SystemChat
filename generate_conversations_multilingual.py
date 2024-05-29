import json
import random
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from tenacity import retry, wait_exponential, stop_after_attempt

languages_file = "multilingual_templates.jsonl"
system_prompts_file = "system_messages.txt"
usecases_file = "user_intents.txt"
out_file = "SystemChat_multilingual2.jsonl"

system_prompts = open(system_prompts_file, "r").readlines()
random.shuffle(system_prompts)
usecases = open(usecases_file, "r").readlines()
random.shuffle(usecases)
languages = open(languages_file, "r").readlines()
random.shuffle(languages)

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

# Create a lock object
lock = threading.Lock()

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

def generate_turn(messages, usecase, language):
    conversation_history = "\n".join([f"Sam: {item['content']}" if item['role'] == 'user' else f"AI: {item['content']}" for item in messages if item['role'] != 'system']) 
    
    metaprompt = language["metaprompt"].format(usecase=usecase, conversation_history=conversation_history)
    response = generate_openai_response(messages=[
            {"role": "user", "content": metaprompt}
        ])

    next_prompt = response.choices[0].message.content.strip()
    
    messages.append({"role": "user", "content": next_prompt})
    
    if "<<||END||>>" in next_prompt:
        return messages
    
    response = generate_openai_response(messages)
    
    messages.append({"role": "assistant", "content": response.choices[0].message.content.strip()})
    
    return messages

def generate_conversation():
    time.sleep(random.randint(1,100))
    while True:
        usecase = random.choice(usecases).strip()
        system_prompt = random.choice(system_prompts).strip()
        language = json.loads(random.choice(languages))
        
        # translate system prompt
        response = generate_openai_response(messages=[
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": f"Please translate from English to {language['language']}:\n\n {system_prompt}"}
        ])
        localized_system_prompt = response.choices[0].message.content.strip()
        
        # translate usecase
        response = generate_openai_response(messages=[
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": f"Please translate from English to {language['language']}:\n\n {usecase}"}
        ])
        localized_usecase = response.choices[0].message.content.strip()

        response = generate_openai_response(
            messages=[
                {"role": "user", "content":language["initialprompt"].format(usecase=localized_usecase)}
            ]
        )
        initial_prompt = response.choices[0].message.content.strip()

        messages = [
            {"role": "system", "content": localized_system_prompt},
            {"role": "user", "content": initial_prompt}
        ]

        response = generate_openai_response(messages)
        initial_response = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": initial_response})

        while "<<||END||>>" not in messages[-1]["content"] and len(messages) < 25:
            messages = generate_turn(messages=messages, usecase=localized_usecase, language=language)
        if messages[-1]["role"] == "user":
            messages.pop()

        print(messages)
        print("\n")
        
        with lock:
            with open(out_file, "a", encoding="utf-8") as f:
                json.dump({"language":language["language"], "usecase":usecase, "system_prompt":system_prompt, "model":model_name, "messages":messages}, f)
                f.write("\n")
    
def main():
    # generate_conversation()
    with ThreadPoolExecutor(max_workers=1000) as executor:
        for _ in range(1000):
            executor.submit(generate_conversation)

if __name__ == "__main__":
    main()
