import json.decoder

import google.generativeai as genai
from utils.enums import LLM
import time


def get_model(api_key, model):
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]
    generation_config = {}
    if model == LLM.GEMINI_1_0_PRO or model == GEMINI_1_0_PRO_001:
        generation_config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 0,
            "max_output_tokens": 2048,
        }
    else:
        generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,  
        }

    model = genai.GenerativeModel(model_name=model,
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    return model

def ask_llm(model: str, batch: list, api_key: str):
    n_repeat = 0
    while True:
        try:
            model = get_model(api_key, model)
            response = model.generate_content(batch)
        except Exception as e:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
            time.sleep(1)
            continue
    return response

