# %%
from openai import AzureOpenAI
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import warnings
from dotenv import load_dotenv
import os
import utils  # Importing the utils module

# %%
load_dotenv()
openai_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("DEPLOYMENT_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL_NAME")
azure_endpoint = os.getenv("AZURE_ENDPOINT")

output_file = "system_messages2.txt"

similarity_threshold = 0.97

system = "You are a helpful assistant that generates predefined system messages for a text-based AI Chatbot. These messages include defining mannerisms, guardrails, functions, roleplaying characters, purposes, behavioral rules, output formats, and specific procedural instructions. If you would refuse to answer for any reason, instead of explaining, simply reply <<||NO||>>."
prompt_template = "A system message is an instruction to a chatbot that comes before any user messages, in order to set the stage for the interaction.  Example: \"you are a helpful AI assistant\" Generate 5 examples of system messages for a text-based AI chatbot in the category '{category}' as a JSON list of strings. Please just the system messages, no explanation or labels."

# %%
client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=openai_api_version,
    azure_endpoint=azure_endpoint
)

# Define categories for system messages
categories = [
    "Mannerisms of the Chatbot",
    "Guardrails of the Chatbot",
    "Functions Available to the Chatbot",
    "Roleplaying Character Definition",
    "Purpose for the Chatbot",
    "Specific Behavioral Rules",
    "Format of the Output",
    "Specific Procedural Instructions"
]

def generate_system_messages(prompt, system, num_responses=5, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=num_responses,
                stop=None,
                temperature=0.7
            )
            valid_messages = [message for choice in response.choices 
                            if "<<||NO||>>" not in choice.message.content 
                            for message in json.loads(choice.message.content.strip())]
            return valid_messages
        except:
            print(f"Rate limit error. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
    raise Exception("Exceeded maximum retries due to rate limit errors")

# Function to run the generation in parallel
def parallel_generate(prompts):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_system_messages, prompt, prompt_template.format(category=category), 5) for category in categories for prompt in [prompt_template.format(category=category)]]
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
    return results

# Generate initial system messages for each category
print("generating initial")
system_messages = []
prompts = [prompt_template.format(category=category) for category in categories]
system_messages.extend(parallel_generate(prompts))
print("system_messages", len(system_messages))

# Deduplicate system messages using the dedupe function from utils.py
unique_system_messages = utils.dedupe(system_messages, model_name="Snowflake/snowflake-arctic-embed-xs", threshold=similarity_threshold)
print("unique_system_messages", len(unique_system_messages))

# Ensure you have at least 1000 unique system messages
while len(unique_system_messages) < 3000:
    print("generating more")
    additional_messages = parallel_generate(prompts)
    unique_system_messages.extend(additional_messages)
    print("before dedupe", len(unique_system_messages))
    unique_system_messages = utils.dedupe(unique_system_messages, model_name="Snowflake/snowflake-arctic-embed-xs", threshold=similarity_threshold)
    print("after dedupe", len(unique_system_messages))

# Verify the number of unique system messages
print(f"Total unique system messages generated: {len(unique_system_messages)}")

# Print a sample of the generated system messages
sample_size = 20
sample_messages = random.sample(unique_system_messages, sample_size)
for message in sample_messages:
    print(message)
    
with open(output_file, 'w') as f:
    for message in unique_system_messages:
        f.write(message)
        f.write("\n")
