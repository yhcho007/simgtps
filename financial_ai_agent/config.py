import yaml
import os


def load_config():
    """Load configuration from config.yml."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# Load configuration
try:
    CONFIG = load_config()
    OPENAI_API_KEY = CONFIG['openai']['api_key']
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    OPENAI_MODEL = CONFIG['openai']['model']
    HUGGINGFACEHUB_API_TOKEN = CONFIG['huggingface']['api_key']
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
except FileNotFoundError as e:
    print(e)
    # Fallback or exit if config file is critical
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o-mini"
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please set it in config.yml or as an environment variable.")
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not HUGGINGFACEHUB_API_TOKEN:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found. Please set it in config.yml or as an environment variable.")

if __name__ == "__main__":
    print(f"Loaded OpenAI API Key: {'*' * 10}")  # Print masked key for security
    print(f"Loaded OpenAI Model: {OPENAI_MODEL}")

