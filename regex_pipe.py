import os
from groq import Groq
import logging
from datetime import datetime
import re
import json

from config import GROQ_API_KEY

logging.basicConfig(
        filename='my.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_prompt(file_path):
    """
    Reads a multi-line prompt from a specified text file.

    Args:
        file_path (str): Path to the text file containing the prompt.

    Returns:
        str: The contents of the text file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def clean_ex(ex):
    bad_chars = ['\n', '<', '>', '\t', '\r', '/', '\\', ':', '*', '?', '"', '|']

    for c in bad_chars:
        ex = ex.replace(c, '')

    return ex

def make_prompt(file_path = 'example.json'):
    empty_prompt = load_prompt('empty_prompt.txt')

    with open(file_path, "r") as file:
        data = json.load(file)

    pos_examples = [empty_prompt]
    for ex in data['activating_examples']:
        pos_examples.append(clean_ex(ex['text']))

    prompt = ",".join(pos_examples)

    logging.info(f"PROMPT: {prompt}")
    return prompt

def extract_regex(text):
    match = re.search(r'`(.*?)`', text)
    if match:
        return match.group(1)  # Extract the captured group
    return None

def get_regex(prompt):
    logging.info("------------------BEGIN GENERATION------------------")
    # client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    client = Groq(api_key=GROQ_API_KEY)

    chat_completion = client.chat.completions.create(
        messages=[
            {   "role": "system",
                "content": "You are a programmer who generates concise regular expressions (regex)."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-70b-8192",
    )

    chat_completion_str = chat_completion.choices[0].message.content

    logging.info(f"COMPLETION: {chat_completion_str}")

    regex = client.chat.completions.create(
        messages=[
            {   "role": "system",
                "content": "You are a proofreader who is an expert at picking out parts of documents."
            },
            {
                "role": "user",
                "content": "extract the final regex from this statement, and output it alone as a single plaintext line with no other content: " + chat_completion_str,
            }
        ],
        model="llama3-70b-8192",
    )

    regex_str = regex.choices[0].message.content

    logging.info(f"PRE FIXED REGEX: {regex_str}")

    final_regex = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "fix this regex string and output it alone as a single plaintext line, ensuring it is enclosed with one pair of backticks `: " + regex_str,
            }
        ],
        model="llama3-70b-8192",
    )

    final_regex_str = final_regex.choices[0].message.content

    logging.info(f"FINAL OUTPUT: {final_regex_str}")

    final_regex_str = extract_regex(final_regex_str)

    logging.info(f"REGEX: {final_regex_str}")

    logging.info("------------------END GENERATION------------------")

    return final_regex_str

def process_regex(regex):
    regex = regex.replace('\x08', '\\b')
    regex = regex.replace('\n', '')
    regex = regex.replace(' ', '')
    #regex = re.escape(regex)
    return regex

def test_regex(regex, test_text):
    logging.info("------------------BEGIN TEST------------------")

    logging.info(f"Test text: {test_text}")
    logging.info(f"Regex: {regex}")

    matches = re.findall(regex, test_text, re.IGNORECASE)

    logging.info(f"RESULT: {matches}")

    logging.info("------------------END TEST------------------")

    return matches

if __name__ == "__main__":
    # regex = get_regex()
    # regex = process_regex(regex)

    # test_text = """
    # Hi there! Welcome to our morning meeting. Good morning, everyone.
    # I wanted to say hello and introduce myself. How are you all doing?
    # Greetings from the marketing team! It's wonderful to meet you all.
    # Nice to see some familiar faces. Good afternoon, ladies and gentlemen.
    # I must head out now - goodbye! Have a good night and sweet dreams.
    # Best wishes for the upcoming holiday. See you all tomorrow!
    # """

    # test_regex(regex=regex, test_text=test_text)   

    prompt = make_prompt(file_path='example.json')
    regex = get_regex(prompt)
    regex = process_regex(regex)

    logging.info(f"PROCESSED REGEX: {regex}")

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

    test_regex(regex, test_text)