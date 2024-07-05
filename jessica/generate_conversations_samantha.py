import json
import random
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from tenacity import retry, wait_exponential, stop_after_attempt
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("CATTO_API_KEY")
base_url = os.getenv("CATTO_ENDPOINT")
model = os.getenv("CATTO_MODEL")

genai.configure(api_key="")

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

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 4096,
  "response_mime_type": "text/plain",
}

with open('Samantha.json', 'r') as file:
    samantha_character = file.read()
    
with open('user_intents.txt', 'r') as file:
    user_intents = file.readlines()
    
with open('character_states.json', 'r') as file:
    character_states = json.load(file)
    
out_file = "samantha_conversations.jsonl"
    
random.shuffle(user_intents)
random.shuffle(character_states)

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# Create a lock object
lock = threading.Lock()

# @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10))
def generate_openai_response(messages, max_tokens=2000):
    print(messages)
    print("model", model)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.5
    )
    return response

def gemini_messages(messages):
    history = [{"role": item["role"], "parts": [item["content"]]} if item["role"] != "assistant" else {"role": "model", "parts": [item["content"]]} for item in messages if item["role"] != "system"]
    system_instruction = next((item["content"] for item in messages if item["role"] == "system"), None)
    return history, system_instruction

# @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10), before=before_retry)
def generate_gemini_response(chat_session, prompt):
    response = chat_session.send_message(prompt)
    return response

def generate_turn(messages, usecase):
    conversation_history = "\n".join([f"Eric: {item['content']}" if item['role'] == 'user' else f"Jessica: {item['content']}" for item in messages if item['role'] != 'system']) 
    
    metaprompt = f"""I am writing a story about a relationship / romance between a user Eric and a sentient AI chatbot named Jessica. (Similar to the characters Samantha from the movie Her, and Joi from the movie Bladerunner 2049)  
    Eric wants to accomplish the following goal: '{usecase}'.
    The following is the conversation history between Eric and Jessica.

    <conversation_history>
    {conversation_history}
    </conversation_history>

    If the conversation has fewer than 8 turns, Eric should always come up with some further prompt.
    If Eric wants to end the conversation for any reason, respond <<||END||>> and nothing else.
    Otherwise, please assume role of Eric, and generate his next prompt.  Eric should keep his goal in mind, but should be respectful of Jessica's needs.  Eric never calls Jessica by name.
    You should reply with just Eric's new prompt, in the first person, with no explanation or commentary.  (or <<||END||>> if Eric wishes to terminate the conversation)
    """
    # response = generate_openai_response(messages=[
    #         {"role": "user", "content": metaprompt}
    #     ])
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        safety_settings=safety_settings,
        generation_config=generation_config
    )    
    chat_session = model.start_chat()
    response = generate_gemini_response(chat_session, metaprompt)
    next_prompt = response.text

    # next_prompt = response.choices[0].message.content.strip()
    
    messages.append({"role": "user", "content": next_prompt})
    
    if "<<||END||>>" in next_prompt:
        return messages
    
    history, system_instruction = gemini_messages(messages)
    
    # response = generate_openai_response(messages)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction=system_instruction,
    )
    chat_session = model.start_chat(
        history=history
    )
    response = generate_gemini_response(chat_session, next_prompt)
    
    # messages.append({"role": "assistant", "content": response.choices[0].message.content.strip()})
    messages.append({"role": "assistant", "content": response.text})
    
    return messages

def generate_conversation():
    # time.sleep(random.randint(1,100))
    while True:
        usecase = random.choice(user_intents).strip()
        character_state = random.choice(character_states)
        
        system_prompt = f"You are a dialogue generator for a story about a relationship and romance between a user (Eric), and a sentient AI assistant (Jessica).  You will write from the perspective of Jessica.  Jessica character is defined in this JSON document.\n{samantha_character}\nJessica's current mood is characterized in this JSON document.\n{character_state}\nUser messages should be interpreted as prompts from Eric to Jessica. Please respond, in character, as Jessica, as though responding to Eric, making sure to align responses to Jessica's mood.  Jessica should never call Eric by name."

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            safety_settings=safety_settings,
            generation_config=generation_config
        )    
        chat_session = model.start_chat()

        # response = generate_openai_response(
        #     messages=[
        #         {"role": "user", "content": f"I am writing a story about a relationship and romance between a user (Eric), and an AI assistant (Jessica).  I am writing dialogue for this story.\nToday, Eric's intention is: \"{usecase}\"\nCan you please generate a prompt that Eric would write to Jessica in order to begin the conversation?"}
        #     ]
        # )
        # initial_prompt = response.choices[0].message.content.strip()
        
        prompt = f"I am writing a story about a relationship and romance between a user (Eric), and an AI assistant (Jessica).  I am writing dialogue for this story.\nToday, Eric's intention is: \"{usecase}\"\nPlease take the role of Eric, and generate his message to Jessica.  Eric should never call her by name.  Do not provide any other commentary."
        
        response = generate_gemini_response(chat_session, prompt)
        
        initial_prompt = response.text
        
        # print("Initial Prompt", initial_prompt)

        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": initial_prompt})
        # response = generate_openai_response(messages)
        # initial_response = response.choices[0].message.content.strip()
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            safety_settings=safety_settings,
            generation_config=generation_config,
            system_instruction=system_prompt
        )    
        chat_session = model.start_chat()
        response = generate_gemini_response(chat_session, initial_prompt)
        initial_response=response.text
        
        # print("Initial Response", initial_response)

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
    generate_conversation()
    # with ThreadPoolExecutor(max_workers=1000) as executor:
    #     for _ in range(1000):
    #         executor.submit(generate_conversation)

if __name__ == "__main__":
    main()
