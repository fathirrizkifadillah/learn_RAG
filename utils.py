import requests
import json
import os
from typing import List, Dict
from groq import Groq

def get_proxy_url():
    """
    Get the Groq proxy URL or default endpoint.
    """
    return os.environ.get('GROQ_BASE_URL', 'https://api.groq.com/')

def get_proxy_headers():
    """
    Get appropriate headers for API calls.
    """
    return {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}"}

def get_groq_key():
    """
    Get the API key (mapped to Groq API key).
    """
    return os.environ.get("GROQ_API_KEY", "")

def map_model(model_name: str) -> str:
    """
    Map Together AI models or other models to appropriate Groq models.
    """
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    if model_name in groq_models or "groq" in model_name.lower():
        return model_name
    # Fallback to a high-performance Groq model
    return "llama-3.3-70b-versatile"

def generate_with_single_input(prompt: str,
                               role: str = 'user',
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str = "llama-3.3-70b-versatile",
                               together_api_key = None,
                               groq_api_key = None,
                               **kwargs):
    """
    Call Groq API with a single text prompt.
    """
    # Map model if necessary
    model = map_model(model)

    # Determine the API key to use
    api_key = groq_api_key or together_api_key or os.environ.get("GROQ_API_KEY", "")

    # Check for valid API key when running locally
    if (not api_key or api_key == "your_groq_api_key_here") and "GROQ_BASE_URL" not in os.environ:
        raise ValueError(
            "Groq API key is missing or set to the placeholder ('your_groq_api_key_here').\n"
            "Please obtain a free API key by signing up at https://console.groq.com/,\n"
            "then open the `.env` file in this directory and replace 'your_groq_api_key_here' with your real API key.\n"
            "After updating `.env`, restart your Jupyter notebook kernel and run the cells again."
        )

    # Remove None parameters for Groq API
    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    # Remove any Together AI specific reasoning parameter if present
    kwargs.pop("reasoning", None)

    # Add other kwargs
    payload.update(kwargs)

    # Call Groq API
    if "GROQ_BASE_URL" in os.environ:
        # Use requests/proxy fallback if GROQ_BASE_URL is set
        url = os.path.join(os.environ["GROQ_BASE_URL"], 'v1/chat/completions')
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.post(url, json=payload, headers=headers, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    else:
        # Use the official Groq client
        client = Groq(api_key=api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()

    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict

def generate_with_multiple_input(messages: List[Dict],
                                 top_p: float = None,
                                 temperature: float = None,
                                 max_tokens: int = 500,
                                 model: str = "llama-3.3-70b-versatile",
                                 together_api_key = None,
                                 groq_api_key = None,
                                 **kwargs):
    """
    Call Groq API with multiple conversation messages.
    """
    # Map model if necessary
    model = map_model(model)

    # Determine the API key to use
    api_key = groq_api_key or together_api_key or os.environ.get("GROQ_API_KEY", "")

    # Check for valid API key when running locally
    if (not api_key or api_key == "your_groq_api_key_here") and "GROQ_BASE_URL" not in os.environ:
        raise ValueError(
            "Groq API key is missing or set to the placeholder ('your_groq_api_key_here').\n"
            "Please obtain a free API key by signing up at https://console.groq.com/,\n"
            "then open the `.env` file in this directory and replace 'your_groq_api_key_here' with your real API key.\n"
            "After updating `.env`, restart your Jupyter notebook kernel and run the cells again."
        )

    # Remove None parameters for Groq API
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    # Remove any Together AI specific reasoning parameter if present
    kwargs.pop("reasoning", None)

    # Add other kwargs
    payload.update(kwargs)

    # Call Groq API
    if "GROQ_BASE_URL" in os.environ:
        # Use requests/proxy fallback if GROQ_BASE_URL is set
        url = os.path.join(os.environ["GROQ_BASE_URL"], 'v1/chat/completions')
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.post(url, json=payload, headers=headers, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    else:
        # Use the official Groq client
        client = Groq(api_key=api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()

    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict
