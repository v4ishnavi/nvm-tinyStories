import openai
import json
import re
import logging
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-3.5-turbo"
evaluation_prompt = """
You are an evaluator of English sentences. Each sentence is scored based on three criteria: 
1. **Consistency**: How logical and coherent the sentence is.
2. **Grammar**: The correctness of grammar, punctuation, and syntax.
3. **Creativity**: The originality and uniqueness of the sentence.
You will provide an integer score out of 10 for each category. Provide brief comments if necessary.

Here is the sentence:
"""

def parse_evaluation(evaluation_text):
    """
    Parse the evaluation text to extract the scores for consistency, grammar, and creativity.
    If a score is not found, it is set to None.

    Args:
    evaluation_text (str): The text containing the evaluation scores.

    Returns:
    tuple: A tuple containing the scores for consistency, grammar, and creativity.
    """
    logging.debug(f"Parsing evaluation text: {evaluation_text}")
    
    consistency_match = re.search(r'Consistency: (\d+)/10', evaluation_text)
    grammar_match = re.search(r'Grammar: (\d+)/10', evaluation_text)
    creativity_match = re.search(r'Creativity: (\d+)/10', evaluation_text)

    logging.debug(f"Extracted Consistency: {consistency_match}")
    logging.debug(f"Extracted Grammar: {grammar_match}")
    logging.debug(f"Extracted Creativity: {creativity_match}")
    
    consistency = int(consistency_match.group(1)) if consistency_match else None
    grammar = int(grammar_match.group(1)) if grammar_match else None
    creativity = int(creativity_match.group(1)) if creativity_match else None
    
    return consistency, grammar, creativity

def evaluate_sentence(sentence):
    """
    Evaluate a sentence based on consistency, grammar, and creativity.
    
    Args:
    sentence (str): The sentence to evaluate.
    
    Returns:
    str: The evaluation response from OpenAI.
    """
    logging.debug(f"Evaluating sentence: {sentence.strip()}")
    prompt = evaluation_prompt + sentence.strip()
    
    # Log the request being made
    logging.debug(f"Prompt being sent to OpenAI: {prompt}")
    
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}]
    )
    
    logging.debug(f"Response received from OpenAI: {response.choices[0].message['content']}")
    return response.choices[0].message['content']

def main():
    logging.info("Loading sentences from file...")
    
    with open("sentences.json", "r") as f:
        sentences = f.readlines()

    results = []
    
    logging.info("Starting sentence evaluations...")
    
    for sentence in sentences:
        logging.info(f"Evaluating sentence: {sentence.strip()}")
        
        evaluation = evaluate_sentence(sentence)
        
        logging.info(f"Evaluation result: {evaluation}")
        
        consistency, grammar, creativity = parse_evaluation(evaluation)
        
        logging.debug(f"Scores -> Consistency: {consistency}, Grammar: {grammar}, Creativity: {creativity}")
        
        result = {
            "sentence": sentence.strip(),
            "consistency": consistency,
            "grammar": grammar,
            "creativity": creativity
        }
        results.append(result)

    logging.info("Saving evaluation results to 'evaluated_sentences.json'...")
    
    with open("evaluated_sentences.json", "w") as outfile:
        json.dump(results, outfile, indent=4)

    logging.info("Evaluations have been saved successfully.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
