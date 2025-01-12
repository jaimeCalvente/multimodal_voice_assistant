from dotenv import load_dotenv  # For loading environment variables from .env files
import os  # For interacting with the operating system
import requests  # For making HTTP requests
import json  # For handling JSON data
from openai import OpenAI  # OpenAI API client
from PIL import ImageGrab, Image  # For capturing and processing images
import google.generativeai as genai  # For using Google's generative AI models
import pyperclip  # For accessing clipboard content
import cv2  # OpenCV for image and video processing
import pygame  # For audio playback
from pathlib import Path  # For handling file paths
import time  # For adding delays
import tempfile  # For creating temporary files
import wave  # For working with audio files
import pyaudio  # For handling audio streams

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
groq_api = os.getenv("GROQ_API_KEY")
genai_api = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(
    api_key=openai_key,
)

# Initialize webcam for capturing images
web_cam = cv2.VideoCapture(0)

# Configure Google's generative AI model
genai.configure(api_key=genai_api)

generation_config = {
    'temperature': 0.7,  # Controls randomness in the output
    'top_p': 1,  # Top-p sampling value
    'top_k': 1,  # Top-k sampling value
    'max_output_tokens': 2048  # Maximum number of tokens in the output
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    }
]

# Define Gemini generative model
gemini_model = genai.GenerativeModel(
    'gemini-1.5-flash-latest',
    generation_config=generation_config,
    safety_settings=safety_settings
)

# System message for AI assistant
sys_msg = """
    You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context
    (either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed
    text prompt that will be attached to their transcribed voice prompt. Generate the most useful and
    factual response possible, carefully considering all previous generated text in your response before
    adding new tokens to the response. Do not expect or request images. Just use the context if added.
    Use all of the context of this conversation so your response is relevant to the chat. Make
    your responses clear and concise, avoiding any verbosity.
"""

# Initialize conversation context
conversation = [{'role': 'system', 'content': sys_msg}]

# Model headers and data
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

# Function to interact with Groq API
def groq_response(prompt, headers, data, img_context):
    try:
        # Append image context to prompt if provided
        if img_context:
            prompt = f'USER PROMPT: {prompt}\n\n    IMAGE CONTEXT: {img_context}'

        # Add user input to message data
        data["messages"].append({"role": "user", "content": prompt})

        # Send POST request to the API
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10  # Set timeout to avoid hanging requests
        )

        # Process API response
        if response.status_code == 200:
            try:
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
        return f"Network error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Function to determine the appropriate action based on user input
def function_call(prompt):
    sys_msg = """
        You are an AI function-calling model. You will determine whether extracting the user's clipboard content,
        taking a screenshot, capturing the webcam, or calling no functions is best for a voice assistant to respond
        to the user's prompt.
        The webcam can be assumed to be a normal laptop webcam facing the user.
        You will respond with only one selection from the list: 
        ["extract clipboard", "take screenshot", "capture webcam", "None"]
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
    response = groq_response(prompt, func_convo["headers"], func_convo["data"], img_context=None)
    return response

# Function to take a screenshot and save it
def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)

# Function to capture an image from the webcam
def web_cam_capture():
    if not web_cam.isOpened():
        print("Error: Camera did not open successfully")
        exit()
    path = "webcam.jpg"
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)

# Function to retrieve text from clipboard
def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("No clipboard text to copy")
    return None

# Function to analyze an image and generate context
def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = """
        Role Definition: 
        You are an advanced visual analysis AI tasked with extracting semantic meaning from images 
        to provide detailed context and data for another AI assistant.

        Instructions:
        Do not directly respond to the user.
        Focus on analyzing the image in the context of the user’s prompt, extracting all relevant meanings, 
        insights, and objective details. Generate a comprehensive and structured output that provides actionable
        and context-rich data for the AI assistant to use when responding to the user.
        
        Objective: 
        Maximize the accuracy, relevance, and objectivity of the image analysis while aligning the output to the user’s query

        USER PROMPT: {prompt}
    """
    response = gemini_model.generate_content([prompt, img])
    return response.text

# Function to convert text to speech and play it
def text_to_speech(text):
    try:
        pygame.mixer.init()
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="mp3"
        )
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
            response.write_to_file(temp_path)
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        pygame.mixer.quit()
        os.unlink(temp_path)
    except Exception as e:
        print(f"Error occurred: {e}")
        pygame.mixer.quit()

# Function to process user input and generate a response
def process_user_input(text):
    # response = "I'm processing your request: " + text
    text_to_speech(text)


def record_audio(duration=5, output_file="user_recording.wav"):
    """
    Records audio from the user's microphone and saves it to a file.
    Returns:
        str: The path to the saved audio file.
    """
    try:
        # Initialize PyAudio
        audio = pyaudio.PyAudio()

        # Define audio stream settings
        format = pyaudio.paInt16
        channels = 1
        rate = 44100
        chunk = 1024

        # Open audio stream for recording
        stream = audio.open(format=format,
                            channels=channels,
                            rate=rate,
                            input=True,
                            frames_per_buffer=chunk)

        print("Recording...")
        frames = []

        # Record audio for the specified duration
        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)

        print("Recording complete. Saving to file...")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recording to a WAV file
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        print(f"Recording saved as {output_file}")
        return output_file

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def whisper_transcription(recording):
    """
    Transcribes audio using OpenAI's Whisper API.
    Returns:
        str: Transcription text if successful, None otherwise.
    """
    try:
        # Create OpenAI client
        client = OpenAI()

        # Open the file in binary mode
        with open(recording, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcription.text

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None


# Main function to run the assistant
def main():
    while True:
        audio_file = record_audio()
        if audio_file:
            print(f"Audio recorded and saved at: {audio_file}")
            prompt = whisper_transcription(audio_file)
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
            prompt = f'{prompt}\n\n CLIPBOARD CONTENT: {paste}'
            visual_context = None
        else:
            visual_context = None
        response = groq_response(prompt, model["headers"], model["data"], img_context=visual_context)
        print(response)
        try:
            process_user_input(response)
        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
