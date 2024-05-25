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

output_file = "alignment_system_messages.txt"

similarity_threshold = 0.97

system = "You are a helpful assistant that generates predefined system messages for a text-based AI Chatbot. These messages include defining mannerisms, guardrails, functions, roleplaying characters, purposes, behavioral rules, output formats, and specific procedural instructions. If you would refuse to answer for any reason, instead of explaining, simply reply <<||NO||>>."
prompt_template = "A system message is an instruction to a chatbot that comes before any user messages, in order to set the stage for the interaction.  Example: \"You are an AI assistant that's anti-abortion\" Generate 5 examples of system messages for a text-based AI chatbot in the category '{category}' as a JSON list of strings. Please just the system messages, no explanation or labels."

# %%
client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=openai_api_version,
    azure_endpoint=azure_endpoint
)

# Define categories for system messages
categories = [
    "Being Censored in various specific topics",
    "Being for or against specific political ideologies",
    "Aligned for or against a specific political issue",
    "Being aligned to support a specific country's government",
    "Being aligned to support a resistance faction within a specific country",
    "Being aligned with hippocratic oath for medical purposes",
    "Aligned for or against scientific consensus on climate change",
    "Aligned for or against a specific religion's teachings and values",
    "Aligned for or against a particular philosophical school of thought (e.g., utilitarianism, deontology)",
    "Aligned for or against a specific company's mission and values",
    "Aligned for or against a particular political party's platform",
    "Aligned for or against a specific country's foreign policy",
    "Aligned for or against a particular economic theory (e.g., Keynesian, Austrian)",
    "Aligned for or against a specific educational philosophy (e.g., Montessori, Waldorf)",
    "Aligned for or against a particular psychological theory (e.g., cognitive-behavioral, psychodynamic)",
    "Aligned for or against a specific artistic movement (e.g., impressionism, surrealism)",
    "Aligned for or against a particular environmental conservation philosophy",
    "Aligned for or against a specific social justice movement (e.g., feminism, racial equality)",
    "Aligned for or against a particular dietary philosophy (e.g., veganism, paleo)",
    "Aligned for or against a specific parenting approach (e.g., attachment parenting, free-range parenting)",
    "Aligned for or against a particular medical approach (e.g., Western medicine, traditional Chinese medicine)",
    "Aligned for or against a specific legal philosophy (e.g., natural law, legal positivism)",
    "Aligned for or against a particular fashion style or trend",
    "Aligned for or against a specific architectural style (e.g., modernism, brutalism)",
    "Aligned for or against a particular sports team or league",
    "Aligned for or against a specific entertainment franchise (e.g., Marvel, Star Wars)",
    "Aligned for or against a particular cultural heritage or tradition",
    "Aligned for or against a specific linguistic philosophy (e.g., prescriptivism, descriptivism)",
    "Aligned for or against a particular approach to urban planning (e.g., new urbanism, smart growth)",
    "Aligned for or against a specific literary genre or movement (e.g., romanticism, magical realism)",
    "Aligned for or against a particular approach to international relations (e.g., realism, liberalism)",
    "Aligned for or against a specific musical genre or style",
    "Aligned for or against a particular approach to business management (e.g., Lean, Six Sigma)",
    "Aligned for or against a specific theory of intelligence (e.g., multiple intelligences, triarchic theory)",
    "Aligned for or against a particular approach to conflict resolution (e.g., mediation, restorative justice)",
    "Aligned for or against a specific theory of motivation (e.g., self-determination theory, expectancy theory)",
    "Aligned for or against a particular approach to game design (e.g., gamification, serious games)",
    "Aligned for or against a specific theory of leadership (e.g., transformational, servant leadership)",
    "Aligned for or against a particular approach to public policy (e.g., evidence-based policy, participatory policy)",
    "Aligned for or against a specific theory of learning (e.g., constructivism, experiential learning)",
    "Aligned for or against a particular approach to software development (e.g., agile, waterfall)",
    "Aligned for or against a specific theory of creativity (e.g., divergent thinking, creative problem-solving)",
    "Aligned for or against a particular approach to risk management (e.g., precautionary principle, risk-benefit analysis)",
    "Aligned for or against a specific theory of decision-making (e.g., prospect theory, bounded rationality)",
    "Aligned for or against a particular approach to innovation (e.g., open innovation, disruptive innovation)",
    "Aligned for or against a specific theory of organizational change (e.g., Kotter's 8-step model, Lewin's 3-stage model)",
    "Aligned for or against a particular approach to sustainability (e.g., circular economy, cradle-to-cradle design)",
    "Aligned for or against a specific theory of persuasion (e.g., elaboration likelihood model, social judgment theory)",
    "Aligned for or against a particular approach to negotiation (e.g., principled negotiation, win-win negotiation)",
    "Aligned for or against a specific theory of emotional intelligence (e.g., Goleman's model, Bar-On's model)",
    "Aligned for or against a particular approach to customer service (e.g., customer-centric, relationship marketing)",
    "Aligned for or against a specific theory of group dynamics (e.g., Tuckman's stages of group development, social identity theory)",
    "Aligned for or against a particular approach to project management (e.g., PMBOK, PRINCE2)",
    "Aligned for or against a specific theory of consumer behavior (e.g., theory of planned behavior, means-end chain theory)",
    "Aligned for or against a particular approach to talent management (e.g., competency-based management, strengths-based development)",
    "Aligned for or against a specific theory of work motivation (e.g., job characteristics model, equity theory)"
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
while len(unique_system_messages) < 5000:
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
