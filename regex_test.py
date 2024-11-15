import re

regex = r"\b(?:hello|greetings|salutations|how are you doing?|it's nice to meet you|how do you do?|good\s+(?:morning|evening|afternoon))\b"

test_text = """

Hello! How are you doing today? I hope you're having a wonderful morning. 

It's so nice to meet you at last. Greetings to you and your family! 

Good afternoon, everyone—I'm glad to be here. How do you do, my dear friend? 

Salutations from our team. Good evening, sir. I'm excited to chat.

Nice to see you here.


On another note, goodbye for now; I must be leaving shortly. 

I’ll see you later! Best wishes on your journey. Good night, sleep well. Until we meet again.

"""

matches = re.findall(regex, test_text, re.IGNORECASE)

print(matches)