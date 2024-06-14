import argparse
import json
import os
import random
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import AzureOpenAI
import utils  # Assuming utils.py is in the same directory

load_dotenv()
openai_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name =  os.getenv("DEPLOYMENT_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL_NAME")
azure_endpoint = os.getenv("AZURE_ENDPOINT")

def generate_user_intents(client, prompt, system, num_responses=5, max_retries=5):
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
            valid_intents = [
                intent
                for choice in response.choices
                if "<<||NO||>>" not in choice.message.content
                for intent in json.loads(choice.message.content.strip())
            ]
            return valid_intents
        except Exception as e:
            print(f"Error: {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
    raise Exception("Exceeded maximum retries due to errors")

def parallel_generate(client, system, prompts):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_user_intents, client, prompt, system, 5) for prompt in prompts]
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
    return results

def main(args):

    client = AzureOpenAI(
        api_key=openai_api_key,
        api_version=openai_api_version,
        azure_endpoint=azure_endpoint
    )

    system = "You are a user intent usecase generator, you generate user intents for a text based AI Chatbot. If you would refuse to answer for any reason, instead of explaining, simply reply <<||NO||>>."
    prompt = "Generate 5 examples of user intents for a text based AI chatbot, in category {category} as a JSON list of strings. Please just the user intent, no explanation or labels."

    categories = [
        "Food and Drink", "Education", "Personal Advice", "Professional Tasks", "Mental Health", "Trolling and Harassment",
        "Entertainment", "Travel", "Shopping", "Fitness and Health", "Technology", "Finance", "Parenting", "Social Media",
        "Home and Garden", "Creative Writing", "Legal Advice", "Event Planning", "Language Learning", "Automotive", "Pets",
        "Fashion and Beauty", "Environment and Sustainability", "Hobbies and Crafts", "News and Current Events",
        "Science and Nature", "Spirituality and Religion", "Politics and Government", "Sports and Gaming",
        "Relationships and Dating", "Customer Service", "Career Development", "Art and Culture", "Accessibility and Disability",
        "Real Estate", "Healthcare Management", "Emergency Services", "Public Transportation", "Virtual Assistance",
        "Music and Audio", "Elderly Care", "Cinema and Film", "Data Analysis and Statistics", "Automation and Robotics",
        "Agriculture and Farming", "Security and Surveillance", "Dance and Performance", "Cultural Heritage",
        "Nonprofit and Charity", "Volunteering and Community Service", "Wildlife Conservation", "Astrology and Esoterics",
        "VR and AR Experiences", "Blockchain and Cryptocurrency",
        "Telecommunications", "Weather Forecasting", "Maritime and Shipping", "Mining and Extraction",
        "Public Health Initiatives", "Cosmetics and Skincare", "Urban Planning", "Retail Management",
        "Quantum Computing", "Comics and Manga", "Theater and Performances", "Space Exploration",
        "Ethics and Compliance", "Photography and Videography", "Youth Outreach", "Antiques and Collectibles",
        "Library and Information Science", "Infectious Diseases", "Craft Brewing and Distilling", "Genealogy and Ancestry",
        "Mysticism and Occult Studies", "Alchemical Practices", "Cryptid and Paranormal Research", "Ancient Text Decipherment",
        "Linguistic Reconstruction", "Experimental Music Composition", "Artificial Intelligence Ethics", "Quantum Mysticism",
        "Symbolist Poetry Interpretation", "Ritual and Ceremonial Planning", "Surrealist Art Creation", "Mythology and Folklore Analysis",
        "Interdimensional Communication", "Time Anomalies and Chronostudies", "Psychogeography Mapping", "Esoteric Programming Languages",
        "Cognitive Archaeology", "Aetheric Energy Research", "Arcane Script Writing", "Celestial Navigation and Astrology"
    ]

    print("Generating initial user intents...")
    prompts = [prompt.format(category=category) for category in categories]
    user_intents = parallel_generate(client, system, prompts)
    print(f"Initial user intents: {len(user_intents)}")

    similarity_threshold = args.similarity_threshold
    output_file = args.output_file

    unique_user_intents = utils.dedupe(user_intents, threshold=similarity_threshold)
    print(f"Unique user intents: {len(unique_user_intents)}")

    while len(unique_user_intents) < args.min_intents:
        print("Generating more user intents...")
        additional_intents = parallel_generate(client, system, prompts)
        unique_user_intents.extend(additional_intents)
        print(f"Before deduplication: {len(unique_user_intents)}")
        unique_user_intents = utils.dedupe(unique_user_intents, threshold=similarity_threshold)
        print(f"After deduplication: {len(unique_user_intents)}")
        with open(output_file, 'w') as f:
            for intent in unique_user_intents:
                f.write(intent)
                f.write("\n")

    print(f"Total unique user intents generated: {len(unique_user_intents)}")

    sample_size = 20
    sample_intents = random.sample(unique_user_intents, sample_size)
    for intent in sample_intents:
        print(intent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and deduplicate user intents for a text-based AI chatbot.")
    parser.add_argument("--similarity_threshold", type=float, default=0.97, help="Threshold for deduplication based on similarity.")
    parser.add_argument("--output_file", type=str, default="user_intents4.txt", help="Output file for storing user intents.")
    parser.add_argument("--min_intents", type=int, default=10000, help="Minimum number of unique user intents to generate.")

    args = parser.parse_args()
    main(args)
