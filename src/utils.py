import openai
import os
import time  # Added missing time import
import tempfile  # Added for better temp file handling
from openai import OpenAI
from dotenv import load_dotenv
import pygame
from pathlib import Path

# Load your OpenAI API key
load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(
    api_key=openai_key,
)

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
    response = "I'm processing your input: " + text
    text_to_speech(response)

# Example usage
if __name__ == "__main__":
    try:
        process_user_input("Hello how are you doing?")
    except Exception as e:
        print(f"Error occurred: {e}")