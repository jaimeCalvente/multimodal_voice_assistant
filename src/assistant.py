from dotenv import load_dotenv
import os
import requests
import json
from openai import OpenAI
from PIL import ImageGrab, Image
import google.generativeai as genai
import pyperclip
import cv2
import pygame
from pathlib import Path
import time  
import tempfile  


# Load environment variables from .env file
load_dotenv()

# Get API key
groq_api = os.getenv("GROQ_API_KEY")
genai_api = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv('OPENAI_API_KEY')

# OpenAI client
client = OpenAI(
    api_key=openai_key,
)


# laptop webcam
web_cam = cv2.VideoCapture(0)

genai.configure(api_key=genai_api)

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',  # Corrected spelling
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',  # Corrected spelling
        'threshold': 'BLOCK_NONE'
    }
]



gemeni_model = genai.GenerativeModel('gemini-1.5-flash-latest',
                                        generation_config= generation_config,
                                        safety_settings=safety_settings)


sys_msg = """
        You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context
        (either a screenshot or a webcam capture). Any phot has already been processed into a highly detailed
        text prompt that will be attached to their transcribed voice prompt. Generate the most useful and
        factual response possible, carefullly considering all previous generated text in your response before
        adding new tokens to the response. Don not expect or request images. just use the context id added
        Use all od the contect of this conversation so your response is relevant to the chat. Make
        your responses clear and concise, avoiding any verbosity
    """
conversation = [{'role': 'system', 'content': sys_msg}]

model = {
    "headers": {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_api}"
    },
    "data": {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ],
        "model": "grok-beta",
        "stream": False,
        "temperature": 0.7
    }
}


def groq_response(prompt, headers, data, img_context):
    try:
        # Modify prompt with image context data
        if img_context:
            prompt = f'USER PROMPT: {prompt}\n\n    IMAGE CONTEXT: {img_context}'

        # add role and content to the message data
        data["messages"].append({"role": "user", "content": prompt})

        # get llm response
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10  # Adding a timeout to prevent hanging
        )

        # Check if the response status code is OK
        if response.status_code == 200:
            try:
                # Attempt to parse the response JSON
                res_msg = response.json()['choices'][0]['message']['content']
                
                if res_msg:
                    conversation.append(res_msg)
                else:
                    return "No response received from model."
                return res_msg
            except (KeyError, json.JSONDecodeError) as e:
                return f"Error parsing response: {str(e)}"
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        # Catch network-related errors
        return f"Network error: {str(e)}"
    except Exception as e:
        # Catch any other unexpected exceptions
        return f"Unexpected error: {str(e)}"


def function_call(prompt):
    sys_msg = """
        You are an AI function-calling model. You will determine whether extracting the user's clipboard content,
        taking a screenshot, capturing the webcam, or calling no functions is best for a voice assistant to respond
        to the user's prompt.
        The webcam can be assumed to be a normal laptop webcam facing the user.
        You will respond with only one selection from the list: 
        ["extract clipboards", "take screenshot", "capture webcam", "None"]
        Do not respond with anything but the most logical selection from that list with no explanations. 
        Format the function call name exactly as listed.    
    """
    func_convo = {
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {groq_api}"
        },
        "data": {
            "messages": [
                {
                    "role": "system",
                    "content": sys_msg
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "grok-beta",
            "stream": False,
            "temperature": 0.7
        }
    }
    # Get response with new data
    response = groq_response(prompt, func_convo["headers"], func_convo["data"], img_context=None)

    return response


def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)


def web_cam_capture():
    if not web_cam.isOpened():
        print("Error: Camera did not open successfully")
        exit()
    path = "webcam.jpg"
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)    


def get_clipboard_text():
    clipboard_content = pyperclip.paste()

    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("No clipboard text to copy")
    return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = """
        You are the vision analysis AI that provides semantic meaning from images to provide context
        to send to another AI that will create a response to the user. Do not respond as the AI assistant
        to the user. Instead take the user prompt input and try to extract all the meaning from the photo
        relevant to the user prompt.
        Then generate as much objective data about the image for the AI
        assistant who will respond to the user. 

        USER PROMPT: {prompt}
    """
    response = gemeni_model.generate_content([prompt, img])

    return response.text

def text_to_speech(text):
    try:
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Get the speech data from OpenAI
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="mp3"
        )
        
        # Create a temporary file in the system's temp directory
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
            response.write_to_file(temp_path)
        
        # Load and play the audio
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        # Clean up
        pygame.mixer.music.unload()
        pygame.mixer.quit()
        
        # Remove the temporary file
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        # Additional cleanup in case of error
        pygame.mixer.quit()

def process_user_input(text):
    # Add processing logic here
    response = "I'm processing your request: " + text
    text_to_speech(response)


def main():
    while True:
        prompt = input('USER: ')
        if prompt.lower() in ['exit', 'quit']:
            break
        # Response from the function call
        call = function_call(prompt)
        
        if 'take screenshot' in call:
            print("Taking screenshot")
            take_screenshot()
            visual_context = vision_prompt(prompt=prompt, photo_path='screenshot.jpg')
        elif 'capture webcam' in call:
            print("Capturing webcam")
            web_cam_capture()
            visual_context = vision_prompt(prompt=prompt, photo_path='webcam.jpg')
        elif 'extract clipboard' in call:
            print("Copying clipboard text")
            paste = get_clipboard_text()
            prompt = f'{prompt}\n\n CLIPBOARD CONTENT : {paste}'
            visual_context = None
        else:
             visual_context = None
        # Regular response
        response = groq_response(prompt, 
                            model["headers"], 
                            model["data"], 
                            img_context=visual_context)
        print(response)
        try:
            process_user_input(response)
        except Exception as e:
            print(f"Error occurred: {e}")



if __name__ == "__main__":
    main()


