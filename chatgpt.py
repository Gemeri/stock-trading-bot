import os
import openai
import sys
from datetime import datetime
import json 

def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: The OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    return api_key

def read_prompt_from_file(file_path="input.txt"):
    if not os.path.exists(file_path):
        print(f"Error: The input file '{file_path}' does not exist.")
        sys.exit(1)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read().strip()
    
    if not prompt:
        print(f"Error: The input file '{file_path}' is empty.")
        sys.exit(1)
    
    return prompt

def load_conversation_history(file_path="history.json"):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return []  # Start with empty history if no file exists

def save_conversation_history(history, file_path="history.json"):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_chatgpt_response(messages, api_key):
    client = openai.OpenAI(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model="o1-preview",  # Using the specified model
            messages=messages,
        )
        return completion.choices[0].message.content  # Accessing the content attribute directly
    except openai.OpenAIError as e:
        print(f"An error occurred while communicating with OpenAI: {e}")
        sys.exit(1)

def write_response_to_file(response, file_path="output.txt"):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response)
        print(f"Response successfully saved to '{file_path}'.")
    except IOError as e:
        print(f"An error occurred while writing to the file '{file_path}': {e}")
        sys.exit(1)

def main():
    api_key = get_api_key()
    prompt = read_prompt_from_file("input.txt")
    history = load_conversation_history("history.json")
    
    if prompt == "del-conv":
        # Delete the conversation history
        if os.path.exists("history.json"):
            os.remove("history.json")
            print("Conversation history deleted.")
        else:
            print("No conversation history to delete.")
        # Optionally, write confirmation to output.txt
        with open("output.txt", 'w', encoding='utf-8') as file:
            file.write("Conversation history deleted.")
        return  # Exit after deleting the conversation history
    
    # Add the user's prompt to the conversation history
    history.append({"role": "user", "content": prompt})
    
    # Get the response from ChatGPT
    response = get_chatgpt_response(history, api_key)
    print("\nChatGPT Response:\n")
    print(response)
    write_response_to_file(response, "output.txt")
    
    # Add the AI's response to the conversation history
    history.append({"role": "assistant", "content": response})
    # Save the updated conversation history
    save_conversation_history(history, "history.json")

if __name__ == "__main__":
    main()