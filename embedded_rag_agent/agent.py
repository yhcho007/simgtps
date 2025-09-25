import json
from models import embedder
from utils import pdf_ocr

class ChatAgent:
    def __init__(self, config_path='config.json'):
        with open(config_path) as f:
            self.config = json.load(f)
        self.backend = self.config['llm_backend']

    def respond(self, text: str) -> str:
        # TODO: add LLM call (transformers or llama.cpp)
        return f"[LLM:{self.backend}] {text}"
