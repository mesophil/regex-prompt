import re

regex = "\b(?:(?:good\s(?:morning|afternoon|evening)|h(?:ello|ow\s(?:(?:are\syou|do\syou\sdo))|(?:(?:it'?s\s)?nice\s+to\s(?:meet|see)\syou|greetings|salutations)))\b[,!]?\s*(?:to\syou|everyone|sir|my\s(?:good\s)?(?:friend|sir))?|hi)(?=[^.!?]*[.!?])"
regex = regex.replace('\x08', '\\b')

#regex = r"\b(?:(?:good\s(?:morning|afternoon|evening))|h(?:ello|ow\s(?:are\s+you|do\s+you\s+do))|(?:it'?s\s)?nice\s+to\s(?:meet|see)\s+you|greetings|salutations)\b[,!?]?\s*(?:to\s+you|everyone|ladies\s+and\s+gentlemen|my\s+(?:good\s+)?(?:friend|sir))?"

test_text = """
Hi there! Welcome to our morning meeting. Good morning, everyone.
I wanted to say hello and introduce myself. How are you all doing?
Greetings from the marketing team! It's wonderful to meet you all.
Nice to see some familiar faces. Good afternoon, ladies and gentlemen.
I must head out now - goodbye! Have a good night and sweet dreams.
Best wishes for the upcoming holiday. See you all tomorrow!
"""

matches = re.findall(regex, test_text, re.IGNORECASE)
print(matches)