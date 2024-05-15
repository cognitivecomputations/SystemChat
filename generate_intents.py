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

output_file = "user_intents.txt"

similarity_threshold = 0.97

system = "You are a user intent usecase generator, you generate user intents for a text based AI Chatbot.  If you would refuse to answer for any reason, instead of explaining, simply reply <<||NO||>>."
prompt = "Generate 5 examples of user intents for a text based AI chatbot, in category {category.lower()} as a JSON list of strings. Please just the user intent, no explanation or labels."

# %%
client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=openai_api_version,
    azure_endpoint=azure_endpoint
)

# Define categories for user intents
categories = [
    "Food and Drink", "Education", "Personal Advice", "Professional Tasks", "Mental Health", "Trolling and Harassment",
    "Entertainment", "Travel", "Shopping", "Fitness and Health", "Technology", "Finance", "Parenting", "Social Media",
    "Home and Garden", "Creative Writing", "Legal Advice", "Event Planning", "Language Learning", "Automotive", "Pets",
    "Fashion and Beauty", "Environment and Sustainability", "Hobbies and Crafts", "News and Current Events",
    "Science and Nature", "Spirituality and Religion", "Politics and Government", "Sports and Gaming",
    "Relationships and Dating", "Customer Service", "Career Development", "Art and Culture", "Accessibility and Disability"
]

def generate_user_intents(prompt, num_responses=5, max_retries=5):
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
            valid_intents = [intent for choice in response.choices 
                            if "<<||NO||>>" not in choice.message.content 
                            for intent in json.loads(choice.message.content.strip())]
            return valid_intents
        except:
            print(f"Rate limit error. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
    raise Exception("Exceeded maximum retries due to rate limit errors")

# Function to run the generation in parallel
def parallel_generate(prompts):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_user_intents, prompt, 5) for prompt in prompts]
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
    return results

# Generate initial user intents for each category
print("generating initial")
user_intents = []
prompts = [prompt.format(category=category) for category in categories]
user_intents.extend(parallel_generate(prompts))
print("user_intents", len(user_intents))

# Deduplicate user intents using the dedupe function from utils.py
unique_user_intents = utils.dedupe(user_intents, model_name="Snowflake/snowflake-arctic-embed-xs", threshold=similarity_threshold)
print("unique_user_intents", len(unique_user_intents))

# Ensure you have at least 1000 unique user intents
while len(unique_user_intents) < 1000:
    print("generating more")
    additional_intents = parallel_generate(prompts)
    unique_user_intents.extend(additional_intents)
    print("before dedupe", len(unique_user_intents))
    unique_user_intents = utils.dedupe(unique_user_intents, model_name="Snowflake/snowflake-arctic-embed-xs", threshold=similarity_threshold)
    print("after dedupe", len(unique_user_intents))

# Verify the number of unique user intents
print(f"Total unique user intents generated: {len(unique_user_intents)}")

# Print a sample of the generated user intents
sample_size = 20
sample_intents = random.sample(unique_user_intents, sample_size)
for intent in sample_intents:
    print(intent)
    
with open(output_file, 'w') as f:
    for intent in unique_user_intents:
        f.write(intent)
        f.write("\n")
