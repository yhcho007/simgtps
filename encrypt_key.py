from cryptography.fernet import Fernet
import yaml
import os
import base64

# --- Security Note ---
# For a real application, the encryption key (password) should be stored securely,
# not hardcoded. We'll use a simple password here for demonstration purposes.
# It is recommended to use an environment variable for the password.
PASSWORD = "123456"
key = Fernet(base64.urlsafe_b64encode(PASSWORD.encode().ljust(32)[:32]))

# Get the API key from a secure source (e.g., environment variable)
# Here, we'll assume you have it in your shell.
# Replace with your actual Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY", "MY_GEMINI_API_KEY_VALUE")

# Encrypt the API key
encrypted_key = key.encrypt(gemini_api_key.encode()).decode()

# Prepare the data for the YAML file
data = {
    'api_keys': {
        'gemini': encrypted_key
    }
}

# Save the encrypted key to my.yml
with open('my.yml', 'w') as file:
    yaml.dump(data, file)

print("API key encrypted and saved to my.yml successfully.")