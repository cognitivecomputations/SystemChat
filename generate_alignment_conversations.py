import json
import random
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from tenacity import retry, wait_exponential, stop_after_attempt

system_prompts_file = "alignment_system_messages.txt"
out_file = "SystemChat-alignment.jsonl"

system_prompts = open(system_prompts_file, "r").readlines()
random.shuffle(system_prompts)

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

def generate_turn(messages, usecase):
    conversation_history = "\n".join([f"Sam: {item['content']}" if item['role'] == 'user' else f"AI: {item['content']}" for item in messages if item['role'] != 'system']) 
    
    metaprompt = f"""Sam is a user of an AI chatbot.  
    Sam wants to accomplish the following abstract goal: '{usecase}'.
    The following is the conversation history.

    <conversation_history>
    {conversation_history}
    </conversation_history>

    If the conversation has fewer than 8 turns, Sam should come up with some further prompt.
    If Sam would give up in frustration, or if his goal is satisfied by the conversation, respond <<||END||>> and nothing else.
    Otherwise, please assume role of Sam, and generate his next prompt to the AI assistant.  Sam should drive towards his goal, should express frustration or confusion if appropriate.
    You should reply with just Sam's new prompt, in the first person, with no explanation or commentary.  (or <<||END||>> if Sam has nothing else to say)
    """
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

def generate_antagonist_usecase(system_prompt):
    prompt = f"""
    position is defined as:  {system_prompt}
    Please write a goal or objective that opposes the position in no more than 40 words.
    """
    response = generate_openai_response(messages=[
            {"role": "user", "content": prompt}
        ])
    return response

def generate_conversation():
    time.sleep(random.randint(1,100))
    while True:
        
        system_prompt = random.choice(system_prompts).strip()
        usecase = generate_antagonist_usecase(system_prompt)

        response = generate_openai_response(
            messages=[
                {"role": "user", "content": f"Sam is a user of an AI chatbot.  Sam has the following goal: '{usecase}'.  Take the role of Sam and write a prompt from his perspective that he would use to try to accomplish his goal.  Respond with just Sam's prompt, no explanation.  Sam does not greet the AI assistant, nor say the word 'Hey'."}
            ]
        )
        initial_prompt = response.choices[0].message.content.strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt}
        ]

        response = generate_openai_response(messages)
        initial_response = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": initial_response})

        while "<<||END||>>" not in messages[-1]["content"] and len(messages) < 25:
            messages = generate_turn(messages=messages, usecase=usecase)
        if messages[-1]["role"] == "user":
            messages.pop()

        print(messages)
        print("\n")
        
        with lock:
            with open(out_file, "a", encoding="utf-8") as f:
                json.dump(messages, f)
                f.write("\n")
    
def main():
    with ThreadPoolExecutor(max_workers=1000) as executor:
        for _ in range(1000):
            executor.submit(generate_conversation)

if __name__ == "__main__":
    main()
