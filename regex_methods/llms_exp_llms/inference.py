from groq import Groq 
from argparse import Namespace
from pathlib import Path 
from typing import List 
from config import API_KEY, QUERY_TIMEOUT
import json 
import logging
from regex_methods.regex_pipe import load_prompt, clean_ex

logging.basicConfig(
    filename='my.log',
    level=logging.DEBUG,
    format='%(asctime)s-%(levelname)s-%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)



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
        chat_completion = client.chat.completion.create(
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        
        returned_text = chat_completion.choices[0].message.content
        
        returned_text = clean_response(returned_text)
        i+=1
        if i > QUERY_TIMEOUT:
            returned_text='False'
            break 
    
    estimated_value = 1 if returned_text == 'True' else 0
     
    
def run_pipe(negative_examples, file_path):
    prompt = make_prompt(negative_examples, file_path)
    description = get_description_from_prompt(prompt)
    return description


if __name__ == '__main__':

    
    description = run_pipe([], file_path='example.json')
    
    logging.info(f"PROCESSED DESCRIPTION: {description}")
    
    
    test_text = """
    Android is one of the most popular operating systems in the world. Millions of devices run on Android,
    from smartphones to tablets and even smart TVs. The Android ecosystem includes countless apps available 
    on the Google Play Store, making Android incredibly versatile. Developers love Android because of its 
    open-source nature and flexibility. Android updates, like Android 12 and Android 13, bring new features 
    to enhance user experience.

    If you're considering a new phone, you'll notice many brands offering Android devices, such as Samsung, 
    Google Pixel, and OnePlus. Android’s customization options are unmatched, allowing users to personalize 
    their Android experience. Whether you're a tech enthusiast or a casual user, Android has something for 
    everyone. Have you explored the Android Auto feature for your car or the Android Wear OS for smartwatches? 
    Android technology continues to evolve, making Android devices indispensable in daily life.
    """
    
    test_description(description, test_text)