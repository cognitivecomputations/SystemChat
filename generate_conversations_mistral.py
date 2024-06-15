import json
import random
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()
api_key = os.getenv("CATTO_API_KEY")
base_url = os.getenv("CATTO_ENDPOINT")
model = os.getenv("CATTO_MODEL")

system_prompts_file = "system_messages.txt"
usecases_file = "user_intents.txt"
out_file = "SystemChatMistral.jsonl"

system_prompts = open(system_prompts_file, "r").readlines()
random.shuffle(system_prompts)
usecases = open(usecases_file, "r").readlines()
random.shuffle(usecases)

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# Create a lock object
lock = threading.Lock()

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10))
def generate_openai_response(messages, max_tokens=2000):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.5
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
    print("next", next_prompt)
    
    messages.append({"role": "user", "content": next_prompt})
    
    if "<<||END||>>" in next_prompt:
        return messages
    
    response = generate_openai_response(messages)

    answer = response.choices[0].message.content.strip()
    print("answer", answer)
    messages.append({"role": "assistant", "content": answer})
    
    return messages

def generate_conversation(id):
    time.sleep(random.randint(1,100))
    while True:
        usecase = random.choice(usecases).strip()
        system_prompt = random.choice([None, random.choice(system_prompts).strip(), random.choice(system_prompts).strip(), random.choice(system_prompts).strip()])

        response = generate_openai_response(
            messages=[
                {"role": "user", "content": f"Sam is a user of an AI chatbot.  Sam wants to accomplish the following abstract task: '{usecase}'.  Invent an instance of this abstract goal, and then take the role of Sam and write a prompt from his perspective that he would use to try to accomplish his goal.  Respond with just Sam's prompt, no explanation.  Sam does not greet the AI assistant, nor say the word 'Hey'."}
            ]
        )
        initial_prompt = response.choices[0].message.content.strip()

        print("initial prompt", initial_prompt)

        messages = []
        if not system_prompt is None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": initial_prompt})
        response = generate_openai_response(messages)
        initial_response = response.choices[0].message.content.strip()
        print("initial response", initial_response)
        messages.append({"role": "assistant", "content": initial_response})

        while "<<||END||>>" not in messages[-1]["content"] and len(messages) < 15:
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
    # Move ThreadPoolExecutor outside the while True loop
    with ThreadPoolExecutor(max_workers=100) as executor:
        while True:
            # Submit new tasks continuously
            futures = [executor.submit(generate_conversation, i) for i in range(100)]
            # Optional: Handle futures if you need to process results or exceptions
            for future in futures:
                try:
                    # Wait for a task to complete if necessary or handle results
                    result = future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

