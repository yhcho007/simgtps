"""Model proxy that supports multiple providers:
- Local model server (HTTP /generate endpoint)
- OpenAI-compatible API (via HTTP requests)

Switch provider via environment variable MODEL_PROVIDER ('local' or 'openai')
This module centralizes calls to the model and handles timeouts/retries.
"""
import os
import requests

MODEL_PROVIDER = os.getenv('MODEL_PROVIDER', 'local')
MODEL_SERVER_URL = os.getenv('MODEL_SERVER_URL', 'http://localhost:9000/generate')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def generate(prompt: str, max_tokens: int = 256) -> str:
    """Generate text based on provider selection."""
    if MODEL_PROVIDER == 'local':
        # Call local model server
        try:
            r = requests.post(MODEL_SERVER_URL, json={'prompt':prompt, 'max_tokens': max_tokens}, timeout=15)
            r.raise_for_status()
            return r.json().get('text', '')
        except Exception as e:
            return 'Local model server error: ' + str(e)
    elif MODEL_PROVIDER == 'openai':
        if not OPENAI_API_KEY:
            return 'OpenAI API key not configured.'
        # Example OpenAI-compatible call (text-generation endpoint). Adjust per vendor API.
        headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
        data = {'prompt': prompt, 'max_tokens': max_tokens}
        try:
            r = requests.post('https://api.openai.com/v1/completions', headers=headers, json={
                'model':'text-davinci-003', 'prompt':prompt, 'max_tokens': max_tokens
            }, timeout=15)
            r.raise_for_status()
            return r.json()['choices'][0]['text']
        except Exception as e:
            return 'Vendor model error: ' + str(e)
    else:
        return 'Unknown MODEL_PROVIDER'
