import json
import logging
import re

from regex_pipe import test_regex

logging.basicConfig(
        filename='my.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def process_activating_examples_single_feature(data) -> str:
    """Extract the most important token (final token) from each activating example and create a regex pattern.
    """
    final_tokens = set()
    
    for ex in data["activating_examples"]:
        final_tokens.add(ex['text'][ex['offsets'][-1]:])

    token_patterns = [re.escape(str(token)) for token in final_tokens]
    regex_pattern = f"({'|'.join(token_patterns)})"

    logging.info(f"Final Tokens: {final_tokens}")
    logging.info(f"Selection Regex: {regex_pattern}")
    
    return regex_pattern

if __name__ == "__main__":
    file_path = 'example.json'

    with open(file_path, "r") as file:
        data = json.load(file)

    regex_pattern = process_activating_examples_single_feature(data)

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

    res = test_regex(regex_pattern, test_text=test_text)