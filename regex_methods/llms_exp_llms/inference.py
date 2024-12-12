from groq import Groq 
from argparse import Namespace
from pathlib import Path 
from typing import List 
#from config import API_KEY, QUERY_TIMEOUT
API_KEY = 7
QUERY_TIMEOUT = 10
import json 
import logging
from regex_methods.regex_pipe import load_prompt, clean_ex

logging.basicConfig(
    filename='my.log',
    level=logging.DEBUG,
    format='%(asctime)s-%(levelname)s-%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def make_prompt_from_tokens(negative_examples, positive_examples):
    empty_prompt = load_prompt('regex_methods/llms_exp_llms/empty_prompt.txt')
    
    full_prompt = [empty_prompt]
    
    positive_example_string = '\n'.join(positive_examples)
    
    negative_example_string = '\n'.join(negative_examples)
    
    full_prompt = '\n'.join([empty_prompt, "Positive Examples:", positive_example_string, "Negative Examples:", negative_example_string])
    return full_prompt
    
    
    


def make_prompt(negative_examples, file_path='example.json'):
    """Makes the prompt out of the positive examples in the given file and any negative examples"""
    empty_prompt = load_prompt('regex_methods/llms_exp_llms/empty_prompt.txt')

    with open(file_path, "r") as file:
        data = json.load(file)

    full_prompt = [empty_prompt]
    for ex in data['activating_examples']:
        full_prompt.append(clean_ex(ex['text']))

    if negative_examples:
        full_prompt.append(". The following list comprises the negative examples: ")

        for n_ex in negative_examples:
            full_prompt.append(clean_ex(n_ex))


    prompt = ",".join(full_prompt)

    logging.info(f"PROMPT: {prompt}")
    return prompt

def clean_response(text):
    start = text.find('{')
    end = text.find('}')
    if start != -1 and end != -1:
        return text[start + 1:end]
    return None

def get_description_from_prompt(prompt, system_prompt_path = "regex_methods/llms_exp_llms/prompt.txt", model='llama-3.3-70b-specdec'):
    logging.info("-------------------------BEGIN GENERATION-------------------------")
    client = Groq(api_key=API_KEY)
    
    
    system_prompt = open(system_prompt_path, 'r').read()
    
    chat_completion = client.chat.completions.create(
        messages = [
            {
                'role': "system",
                "content": system_prompt
            },
            {
                'role': 'user',
                'content': prompt
            }
            
        ],
        model = model       
    )
    
    returned_description = chat_completion.choices[0].message.content
    
    returned_description = clean_response(returned_description)
    
    logging.info(f"FINAL OUTPUT: {returned_description}")
    
    
    return returned_description




def test_description(description, test_paragraph, system_prompt_path='regex_methods/llms_exp_llms/eval_prompt.txt', model='llama-3.3-70b-specdec'):
    
    logging.info("-------------------------TESTING LLMS EXPLAIN LLMS-------------------------")
    
    system_prompt = open(system_prompt_path, 'r').read()
    
    prompt = ["Description:", description, "Paragraph for Evaluation: ", test_paragraph ]
    
    prompt = '\n'.join(prompt)
    
    client = Groq(api_key = API_KEY)
    returned_text=""
    i = 0
    while not (returned_text == 'True' or returned_text =='False'):
        chat_completion = client.chat.completions.create(
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            model=model
        )
        
        returned_text = chat_completion.choices[0].message.content
        
        returned_text = clean_response(returned_text)
        i+=1
        if i > QUERY_TIMEOUT:
            returned_text='False'
            break 
    
    estimated_value = returned_text == 'True'
    return estimated_value
     
    
def run_pipe(negative_examples, file_path, verbose=True):
    prompt = make_prompt(negative_examples, file_path)
    if verbose:
        print(prompt)
    description = get_description_from_prompt(prompt)
    if verbose:
        print(description)
    return description


if __name__ == '__main__':

    
    description = run_pipe([], file_path='example.json')
    
    logging.info(f"PROCESSED DESCRIPTION: {description}")
    
    
    test_text = """
    Hey, how are you doing today, there, how is your day going. It is good seeing you
    """
    
    prediction = test_description(description, test_text)
    print(prediction)