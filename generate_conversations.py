import json
import random
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import threading

system_prompts_file = "mechanical_system_messages.txt"
usecases_file = "user_intents.txt"
out_file = "SystemChat.jsonl"

system_prompts = open(system_prompts_file, "r").readlines()
random.shuffle(system_prompts)
usecases = open(usecases_file, "r").readlines()
random.shuffle(usecases)

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
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "user", "content": metaprompt}
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7
    )
    next_prompt = response.choices[0].message.content.strip()
    
    messages.append({"role": "user", "content": next_prompt})
    
    if "<<||END||>>" in next_prompt:
        return messages
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    messages.append({"role": "assistant", "content": response.choices[0].message.content.strip()})
    
    return messages

def generate_conversation():
    while True:
        usecase = random.choice(usecases).strip()
        system_prompt = random.choice(system_prompts).strip()

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": f"Sam is a user of an AI chatbot.  Sam wants to accomplish the following abstract task: '{usecase}'.  Invent an instance of this abstract goal, and then take the role of Sam and write a prompt from his perspective that he would use to try to accomplish his goal.  Respond with just Sam's prompt, no explanation.  Sam does not greet the AI assistant, nor say the word 'Hey'."}
            ],
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7
        )
        initial_prompt = response.choices[0].message.content.strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt}
        ]

        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.7
        )
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
    with ThreadPoolExecutor(max_workers=50) as executor:
        for _ in range(50):
            executor.submit(generate_conversation)

if __name__ == "__main__":
    main()
