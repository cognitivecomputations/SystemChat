import json
import random
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from tenacity import retry, wait_exponential, stop_after_attempt

system_prompts_file = "system_messages_arabic.txt"
usecases_file = "user_intents_arabic.txt"
out_file = "SystemChatArabic.jsonl"

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
    conversation_history = "\n".join([f"سام: {item['content']}" if item['role'] == 'user' else f"ذكاء اصطناعي: {item['content']}" for item in messages if item['role'] != 'system']) 
    
    metaprompt = f"""سام هو مستخدم لروبوت الدردشة الذكي.  
سام يريد تحقيق الهدف التالي المجرد: '{usecase}'.
    التاريخ التالي هو سجل المحادثة.

    <سجل المحادثة>
    {conversation_history}
    </سجل المحادثة>

    إذا كانت المحادثة تحتوي على أقل من 8 جولات، يجب على سام أن يأتي بمزيد من الموجهات.
    إذا كان سام سيتخلى عن المحادثة بسبب الإحباط، أو إذا كان هدفه قد تحقق من المحادثة، قم بالرد بـ <<||END||>> ولا شيء آخر.
    خلاف ذلك، يرجى افتراض دور سام وتوليد موجهه التالي لمساعد الذكاء الاصطناعي. يجب على سام أن يتجه نحو هدفه، ويعبر عن إحباطه أو ارتباكه إذا كان ذلك مناسباً.
    يجب أن ترد فقط بموجه سام الجديد، بصيغة المتكلم، بدون أي تفسير أو تعليق. (أو <<||END||>> إذا لم يكن لدى سام شيء آخر ليقوله)
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

def generate_conversation():
    time.sleep(random.randint(1,100))
    while True:
        usecase = random.choice(usecases).strip()
        system_prompt = random.choice(system_prompts).strip()

        response = generate_openai_response(
            messages=[
                {"role": "user", "content": f"سام هو مستخدم لروبوت الدردشة الذكي. سام يريد تحقيق الهدف المجرد التالي: '{usecase}'. اختر مثالاً لهذا الهدف المجرد، ثم خذ دور سام واكتب موجهًا من منظوره الذي سيستخدمه لمحاولة تحقيق هدفه. رد فقط بموجه سام، بدون أي تفسير. سام لا يحيي مساعد الذكاء الاصطناعي، ولا يقول كلمة 'مرحباً'."}
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
    # generate_conversation()
    with ThreadPoolExecutor(max_workers=1000) as executor:
        for _ in range(1000):
            executor.submit(generate_conversation)

if __name__ == "__main__":
    main()
