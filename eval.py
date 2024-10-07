import json
import re
import logging
import os
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI
load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY') 
)

MODEL = "gpt-3.5-turbo"
evaluation_prompt_template = """
The following exercise tests the student’s language abilities and creativity. 
The student is given a beginning of a story and is required to complete it.

Please evaluate the part written by the student after the "***" symbol in terms of **grammar**, **creativity**, and **consistency** with the given prompt.

Here is the student's prompt and completion:
{}

Your task is to assess the following:
- **Grammar**: Are there any grammatical errors?
- **Consistency**: Does the student’s completion logically follow from the beginning?
- **Creativity**: How creative or original is the student's addition to the story?

Give a score out of 10 for each:
- Consistency: X/10
- Grammar: X/10
- Creativity: X/10

Additionally, please provide an estimated age group based on the student’s completion (options: A: 3 or under, B: 4-5, C: 6-7, D: 8-9, E: 10-12):
"""

async def evaluate_sentence(sentence):
    """
    Evaluate a sentence based on consistency, grammar, and creativity using asynchronous API calls.
    
    Args:
    sentence (str): The sentence to evaluate.
    
    Returns:
    dict: A dictionary containing the evaluation and age group.
    """
    logging.debug(f"Evaluating sentence: {sentence.strip()}")
    prompt = evaluation_prompt_template.format(sentence.strip())

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}]
    )

    eval_text = response.choices[0].message.content

    logging.debug(f"Evaluation response: {eval_text}")
    return eval_text


def parse_evaluation(evaluation_text):
    """
    Parse the evaluation text to extract scores for consistency, grammar, and creativity, 
    and determine the age group.
    """
    logging.debug(f"Parsing evaluation text: {evaluation_text}")
    
    # Use regex to extract scores
    consistency_match = re.search(r'[Cc]onsistency:.*?(\d+)/10', evaluation_text)
    grammar_match = re.search(r'[Gg]rammar:.*?(\d+)/10', evaluation_text)
    creativity_match = re.search(r'[Cc]reativity:.*?(\d+)/10', evaluation_text)

    consistency = int(consistency_match.group(1)) if consistency_match else None
    grammar = int(grammar_match.group(1)) if grammar_match else None
    creativity = int(creativity_match.group(1)) if creativity_match else None
    age_group_match = re.search(r'[Aa]ge group:.*?([A-E]|[0-9]+[-]?[0-9]*)', evaluation_text)
    
    if age_group_match:
        # print("age found")
        age_group = age_group_match.group(1)
        if '-' in age_group: 
            if '10' in age_group or '12' in age_group:
                age_group = 'E'
            elif '8' in age_group or '9' in age_group:
                age_group = 'D'
            elif '6' in age_group or '7' in age_group:
                age_group = 'C'
            elif '4' in age_group or '5' in age_group:
                age_group = 'B'
            else:
                age_group = 'A'
    else:
        age_group = None  

    return consistency, grammar, creativity, age_group


async def main():
    logging.info("Loading generated stories from file...")
    
    with open("sentences.json", "r") as f:
        stories = json.load(f)

    results = []
    
    logging.info("Starting sentence evaluations...")
    
    for story_data in stories:
        story = story_data['story']
        logging.info(f"Evaluating story: {story.strip()}")
        
        evaluation = await evaluate_sentence(story) 
        
        logging.info(f"Evaluation result: {evaluation}")
        
        consistency, grammar, creativity, age_group = parse_evaluation(evaluation)
        
        result = {
            "story": story,
            "consistency": consistency,
            "grammar": grammar,
            "creativity": creativity,
            "age_group": age_group
        }
        results.append(result)

    logging.info("Saving evaluation results to 'evaluated_stories.json'...")
    
    with open("evaluated_stories.json", "w") as outfile:
        json.dump(results, outfile, indent=4)

    logging.info("Evaluations have been saved successfully.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())
