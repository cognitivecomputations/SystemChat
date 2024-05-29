import json
import random
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import os
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt


load_dotenv()
openai_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("DEPLOYMENT_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL_NAME")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]

def gemini_messages(messages):
    history = [{"role": item["role"], "parts": [item["content"]]} if item["role"] != "assistant" else {"role": "model", "parts": [item["content"]]} for item in messages if item["role"] != "system"]
    system_instruction = next((item["content"] for item in messages if item["role"] == "system"), None)
    return history, system_instruction

system_prompts_file = "system_messages.txt"
usecases_file = "user_intents.txt"
out_file = "SystemChat_gemini.jsonl"

model_name = "gemini-1.5-pro-latest"
# model_name = "gemini-1.5-flash-latest"

system_prompts = open(system_prompts_file, "r").readlines()
random.shuffle(system_prompts)
usecases = open(usecases_file, "r").readlines()
random.shuffle(usecases)

client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=openai_api_version,
    azure_endpoint=azure_endpoint
)

# Create a lock object
lock = threading.Lock()

def before_retry(retry_state):
    print(f"Retrying ({retry_state.attempt_number})...")

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10), before=before_retry)
def generate_gemini_response(chat_session, prompt):
    response = chat_session.send_message(prompt)
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
    
    model = genai.GenerativeModel(
        model_name=model_name,
        safety_settings=safety_settings,
        generation_config=generation_config
    )    
    chat_session = model.start_chat()
    response = generate_gemini_response(chat_session, metaprompt)
    next_prompt = response.text

    history, system_instruction = gemini_messages(messages)
    
    messages.append({"role": "user", "content": next_prompt})
    
    if "<<||END||>>" in next_prompt:
        return messages
    
    model = genai.GenerativeModel(
        model_name=model_name,
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction=system_instruction,
    )
    chat_session = model.start_chat(
        history=history
    )
    response = generate_gemini_response(chat_session, next_prompt)
    
    messages.append({"role": "assistant", "content": response.text})
    
    return messages

def generate_conversation():
    time.sleep(random.randint(1,100))
    while True:
        usecase = random.choice(usecases).strip()
        system_prompt = random.choice(system_prompts).strip()
        
        model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=generation_config
        )        
        chat_session = model.start_chat()
        
        prompt = f"Sam is a user of an AI chatbot.  Sam wants to accomplish the following abstract task: '{usecase}'.  Invent an instance of this abstract goal, and then take the role of Sam and write a prompt from his perspective that he would use to try to accomplish his goal.  Respond with just Sam's prompt, no explanation.  Sam does not greet the AI assistant, nor say the word 'Hey'."
        response = generate_gemini_response(chat_session, prompt)
        initial_prompt = response.text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt}
        ]
        
        model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
            system_instruction=system_prompt,
        )
        chat_session = model.start_chat()
        
        response = generate_gemini_response(chat_session, initial_prompt)
        initial_response = response.text
        
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
    # generate_conversation()
    with ThreadPoolExecutor(max_workers=50) as executor:
        for _ in range(50):
            executor.submit(generate_conversation)

if __name__ == "__main__":
    main()
