"""!pip install scipy
!pip install tenacity
!pip install tiktoken
!pip install termcolor 
!pip install openai
!pip install requests
!pip install dotenv"""

import json
from openai import OpenAI
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import time
import os
from dotenv import load_dotenv

GPT_MODEL = "gpt-3.5-turbo-1106"


# Load the .env file
path = './environment.env'
load_dotenv(dotenv_path=path)
api_key = os.getenv("OPENAI_KEY")

functions = [

    #Use GPT and a single function with mutliple parameters to turn lights on. Ask User for additional information if needed or use other funcions.
    {
        "name": "turn_lights_on",
        "description": "Accepts parameters and turns lights on. Requires the area (room), color, and intensity.",
        "parameters": {
            "type": "object",
            "properties": {
                "area": {
                    "type": "string",
                    "description": "The area where the lights should be turned on. Allowed values: 'living room', 'kitchen', 'bedroom', 'bathroom', 'garage'."
                },
                "color": {
                    "type": "string",
                    "description": "Color of the lights. Allowed values: 'white', 'red', 'blue', 'green', 'yellow'."
                },
                "intensity": {
                    "type": "number",
                    "description": "Intensity of the lights on a scale of 1 to 5."
                }
            },
            "required": ["area", "color", "intensity"]
        }
    },
    {
        "name": "check_location",
        "description": "Use this function to check for the location of the User",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["Check"], "description": "Send this request to get motion sensor data."}
            },
            "required": ["action"]
        }
    },
    #GPT Can update any type of calendar Event.
    {
        "name": "calendar_update",
        "description": "Add, update, remove, or manage activities in the calendar. Ask User for missing parameters if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "update", "remove"], "description": "Action to perform on the calendar"},
                "type": {"type": "string", "enum": ["reminder", "birthday", "meeting"], "description": "Type of calendar event"},
                "content": {"type": "string", "description": "Content or details of the calendar event"}
            },
            "required": ["action", "type", "content"]
        }
    },
    #GPT could get an alert from detection of movement in video, then send the short video to Azure Video and get a description - then provide the User with a voice synthesized description of the Event.
    {
        "name": "analyize_video",
        "description": "When movement is detected, use this function to send API call to Video Analyzer to create description of event.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_location": {"type": "string", "description": "Cloud location of video."},
                "date_time": {"type": "string", "description": "Event datetime"},
            },
            "required": ["video_location", "date_time"]
        }
    },
    #By indexing videos, User can "search" and Event in the video.
    {
        "name": "search_video",
        "description": "Search through indexed videos using a query and optional date",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for the video"},
                "date": {"type": "string", "description": "Optional date for narrowing down the search"}
            },
            "required": ["query"]
        }
    }
]

client = OpenAI(api_key=api_key)

system = """You are an AI Intelligent Home Assistant. You answer User queries, execute requests and monitor house systems using functions, you can control systems with APIs. You may ask clarifying questions to User if needed."""
def show_json(obj):
    display(json.loads(obj.model_dump_json()))

# Conversation history
conversation_history = []

# Main function to run the conversation loop
def run_conversation():
    global conversation_history

    while True:
        try:
            # Get user input
            user_input = input("You: ")
            if user_input.lower() == 'esc':
                break

            # Append user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Create OpenAI API call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": system},
                    *conversation_history
                ],
                functions=functions  # Assuming 'functions' is defined elsewhere
            )

            # Process response
            response_data = json.loads(response.model_dump_json())['choices'][0]['message']

            # Handle function call or content response
            if response_data.get('content'):
                # Display content response
                print(f"Assistant: {response_data['content']}")
                # Update conversation history with assistant's content response
                conversation_history.append({"role": "assistant", "content": response_data['content']})
            elif response_data.get('function_call'):
                # Handle function call
                function_call = response_data['function_call']
                function_call_display = f"Function Call: {function_call['arguments']}"
                print(function_call_display)
                # Update conversation history with assistant's function call response
                conversation_history.append({"role": "assistant", "content": function_call_display})

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

# Run the conversation loop
run_conversation()