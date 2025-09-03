import requests
import json
from typing import Dict, Any


class AIClient:
    """
    Client for the personal AI assistant running on FastAPI.
    """

    def __init__(self, host: str = "http://127.0.0.1", port: int = 8125):
        self.url = f"{host}:{port}/chat"

    def send_message(self, message: str) -> Dict[str, Any]:
        """
        Sends a message to the AI assistant and returns the JSON response.

        Args:
            message: The user's input message string.

        Returns:
            A dictionary containing the AI assistant's response.
        """
        payload = {"message": message}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to the server: {e}")
            return {"response": "Could not connect to the AI assistant. Please check if the server is running."}

    def chat(self):
        """
        Starts an interactive chat session with the AI assistant.
        """
        print("Starting a chat session with your AI assistant.")
        print("Type 'exit' to end the conversation.")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Ending chat session. Goodbye!")
                break

            response = self.send_message(user_input)
            print(f"AI Assistant: {response.get('response', 'No response.')}")


if __name__ == "__main__":
    # Create an instance of the client
    client = AIClient()

    # Start the interactive chat session
    client.chat()