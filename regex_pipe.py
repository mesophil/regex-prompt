import os
from groq import Groq
import logging
from datetime import datetime
import re

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

def get_regex():
    logging.info("------------------BEGIN GENERATION------------------")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    prompt = load_prompt("prompt.txt")

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
        model="llama-3.2-90b-vision-preview",
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
        model="gemma2-9b-it",
    )

    regex_str = regex.choices[0].message.content

    final_regex = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "fix this regex string and output it alone as a single plaintext line with no other content: " + regex_str,
            }
        ],
        model="gemma2-9b-it",
    )

    final_regex_str = final_regex.choices[0].message.content

    logging.info(f"REGEX: {final_regex_str}")

    logging.info("------------------END GENERATION------------------")

    return final_regex_str

def process_regex(regex):
    regex = regex.replace('\x08', '\\b')
    regex = regex.replace('\n', '')
    regex = regex.replace(' ', '')
    return regex

def test_regex(regex, test_text):
    logging.info("------------------BEGIN TEST------------------")

    logging.info(f"Test text: {test_text}")
    logging.info(f"Regex: {regex}")

    print(regex)

    matches = re.findall(re.escape(regex), test_text, re.IGNORECASE)

    logging.info(f"RESULT: {matches}")

    logging.info("------------------END TEST------------------")

    return matches

if __name__ == "__main__":
    regex = get_regex()
    regex = process_regex(regex)

    test_text = """
    Hi there! Welcome to our morning meeting. Good morning, everyone.
    I wanted to say hello and introduce myself. How are you all doing?
    Greetings from the marketing team! It's wonderful to meet you all.
    Nice to see some familiar faces. Good afternoon, ladies and gentlemen.
    I must head out now - goodbye! Have a good night and sweet dreams.
    Best wishes for the upcoming holiday. See you all tomorrow!
    """

    test_regex(regex=regex, test_text=test_text)   