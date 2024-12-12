from groq import Groq
from openai import OpenAI
from config import GROQ_API_KEY, ML_API_KEY, ML_URL, MODEL_NAME

def get_client(client):
    if client == 'Lev':
        return OpenAI(base_url = ML_URL, api_key=ML_API_KEY)
    elif client == 'Groq':
        return Groq(api_key=GROQ_API_KEY)
    else:
        raise NotImplementedError(f"API {client} not implemented yet")