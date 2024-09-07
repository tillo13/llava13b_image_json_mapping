import os
import json
import re
import time
from datetime import datetime
from PIL import Image
from utilities.ollama_utils import (
    install_and_setup_ollama,
    kill_existing_ollama_service,
    clear_gpu_memory,
    start_ollama_service_windows,
    stop_ollama_service,
    is_windows,
    get_story_response_from_model
)
from utilities.standard_image_detection_utils import generate_face_profile
from utilities.image_utils import zoom_out_and_pad
import atexit

# GLOBAL VARIABLES section
IMAGES_DIR = "images"  # Directory where images are stored
JSON_FILE_LOCATION = "json_profiles"
MODEL_NAME = "llava:13b"

# Create output directory if it does not exist
if not os.path.exists(JSON_FILE_LOCATION):
    os.makedirs(JSON_FILE_LOCATION)

# Function to clean response
def clean_response(response):
    response = re.sub(r"^.*?(yes|no|male|female|blue|green|brown|hazel|gray|blonde|brunette|black|red|(\d{1,3})|white|yellow|brown|black|tan|olive|pale|swimwear|shirt|jacket|pants|shorts|plain|striped|checked|polka-dot)\b.*$", r"\1", response, flags=re.IGNORECASE)
    return response.strip()

# Function to get certainty for model responses
def get_certainty(instruction, answer):
    CERTAINTY_PROMPT_TEMPLATE = "On a scale of 1-100, how certain are you about the answer '{answer}' to the question '{question}'? Respond with just a number."
    certainty_instruction = CERTAINTY_PROMPT_TEMPLATE.format(question=instruction, answer=answer)
    certainty_response = get_story_response_from_model(MODEL_NAME, certainty_instruction)
    clean_certainty_response = clean_response(certainty_response)
    print(f"\nCertainty Question: {certainty_instruction}\nCertainty: {clean_certainty_response}")
    return clean_certainty_response

# Function to generate image description
def generate_image_description(image_path, prompt):
    try:
        result = get_story_response_from_model(MODEL_NAME, prompt)
    except Exception as e:
        raise
    return result

# Function to handle common fallback response checks and replacements
def preprocess_response(response, fallback_value="Unknown"):
    default_responses = [
        "I'm sorry", "I am not able to see images",
        "privacy", "general information", "I do not have personal opinions",
        "only respond with descriptive features"
    ]
    for default_response in default_responses:
        if default_response in response:
            return fallback_value
    return response

# Main process image function
def process_image(image_path):
    print(f"Processing image: {image_path}")
    try:
        # Try generating face profile
        face_profile = generate_face_profile(image_path)
    except ValueError as e:
        if "No face detected" in str(e):
            print(f"No face detected in {image_path}. Trying to zoom out and add padding...")
            new_image_path = zoom_out_and_pad(image_path)
            print(f"Reprocessing with zoomed-out image: {new_image_path}")
            face_profile = generate_face_profile(new_image_path)
        else:
            raise e

    # Create output file name based on the input image file
    output_file_name = f"{os.path.basename(image_path).replace('.', '_')}.json"
    json_file = os.path.join(JSON_FILE_LOCATION, output_file_name)

    left_eye_color_guess = face_profile["physical_features"]["left_eye_color_guess"]
    right_eye_color_guess = face_profile["physical_features"]["right_eye_color_guess"]
    facial_hair_color_guess = face_profile["physical_features"]["facial_hair"]["color_guess"]
    head_hair_color_guess = face_profile["physical_features"]["head_hair"]["color_guess"]

    instructions = {
        "description": "Describe this image. ONLY respond with descriptive features. No extra texts.",
        "wearing_hat": "Is this person wearing a hat? Respond with only 'yes' or 'no'.",
        "eye_color": "What is the eye color of this person? Provide only one of these colors: blue, green, brown, hazel, gray.",
        "hair_color": "What color is the person's hair? Provide only one of these colors: blonde, brunette, black, red or specify 'none'.",
        "facial_hair_color": "What color is the person's facial hair, if any? Provide only one of these colors: blonde, brunette, black, red or 'none'.",
        "pose": "What pose is the person in? Provide a single word: front, side, or back.",
        "age_estimation": "Estimate this person's age in years. Provide only the number.",
        "gender": "What is the gender of this person? Respond with only 'male' or 'female'.",
        "skin_tone": "What is the skin tone of this person? Provide one of these: white, yellow, brown, black, tan, olive, pale.",
        "wearing_glasses": "Is this person wearing glasses? Respond with only 'yes' or 'no'.",
        "upper_body_visible": "Is the person's upper body visible? Respond with only 'yes' or 'no'.",
        "lower_body_visible": "Is the person's lower body visible? Respond with only 'yes' or 'no'."
    }

    descriptions = {}
    for key, instruction in instructions.items():
        response = generate_image_description(image_path, instruction)
        clean_response_text = preprocess_response(response, fallback_value="Unknown")
        descriptions[instruction] = clean_response(clean_response_text)

    profile = {
        "metadata": {
            "filename": image_path,
            "file_location": os.path.abspath(image_path)
        },
        "body_structure": {
            "pose": descriptions[instructions["pose"]]
        },
        "head": {
            "facial_landmarks": {
                "eyes": face_profile["facial_landmarks"]["eyes"],
                "nose": face_profile["facial_landmarks"]["nose"],
                "mouth": face_profile["facial_landmarks"]["mouth"]
            },
            "physical_features": {
                "eyes": {
                    "color": descriptions[instructions["eye_color"]],
                    "llava13b_certainty": get_certainty(instructions["eye_color"], descriptions[instructions["eye_color"]]),
                    "left_eye_color": face_profile["physical_features"]["left_eye_color"],
                    "right_eye_color": face_profile["physical_features"]["right_eye_color"],
                    "left_eye_color_guess": left_eye_color_guess,
                    "right_eye_color_guess": right_eye_color_guess
                },
                "facial_hair": {
                    "standard_color": face_profile["physical_features"]["facial_hair"]["color"],
                    "standard_guess": facial_hair_color_guess,
                    "llava13b_guess": descriptions[instructions["facial_hair_color"]],
                    "llava13b_certainty": get_certainty(instructions["facial_hair_color"], descriptions[instructions["facial_hair_color"]])
                },
                "head_hair": {
                    "standard_color": face_profile["physical_features"]["head_hair"]["color"],
                    "standard_guess": head_hair_color_guess,
                    "llava13b_guess": descriptions[instructions["hair_color"]],
                    "llava13b_certainty": get_certainty(instructions["hair_color"], descriptions[instructions["hair_color"]])
                },
                "skin_tone": {
                    "llava13b_guess": descriptions[instructions["skin_tone"]],
                    "llava13b_certainty": get_certainty(instructions["skin_tone"], descriptions[instructions["skin_tone"]])
                }
            },
            "wearing_hat": {
                "present": "yes" in descriptions[instructions["wearing_hat"]].lower(),
                "llava13b_color_guess": descriptions[instructions["wearing_hat"]],
                "llava13b_certainty": get_certainty(instructions["wearing_hat"], descriptions[instructions["wearing_hat"]])
            },
            "gender": {
                "value": descriptions[instructions["gender"]],
                "llava13b_certainty": get_certainty(instructions["gender"], descriptions[instructions["gender"]])
            }
        },
        "accessories": {
            "glasses": {
                "present": "yes" in descriptions[instructions["wearing_glasses"]].lower(),
                "type": None,
                "color": None,
                "llava13b_certainty": get_certainty(instructions["wearing_glasses"], descriptions[instructions["wearing_glasses"]])
            }
        },
        "clothing": {
            "upper_body": {
                "type": None,
                "color": None,
                "pattern": None
            },
            "lower_body": {
                "type": None,
                "color": None
            }
        },
        "contextual_features": {
            "height": None,
            "build": None,
            "movement": None
        },
        "description": descriptions[instructions["description"]],
        "age_estimation": {
            "value": descriptions[instructions["age_estimation"]],
            "llava13b_certainty": get_certainty(instructions["age_estimation"], descriptions[instructions["age_estimation"]])
        },
        "reference_images": face_profile["reference_images"]
    }

    with open(json_file, 'w') as f:
        json.dump(profile, f, indent=2)

    print(f"Generated JSON file: {json_file}")
    for instruction, description in descriptions.items():
        clean_desc = clean_response(description)
        print(f"\nInstruction: {instruction}\nDescription: {clean_desc}")

def main():
    kill_existing_ollama_service()
    clear_gpu_memory()

    install_and_setup_ollama(MODEL_NAME)
    
    if is_windows():
        service_started = start_ollama_service_windows()
        if not service_started:
            print("Ollama service failed to start. Exiting.")
            return

    for filename in os.listdir(IMAGES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_path = os.path.join(IMAGES_DIR, filename)
                process_image(image_path)
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

    stop_ollama_service()
    clear_gpu_memory()

if __name__ == "__main__":
    atexit.register(stop_ollama_service)
    atexit.register(clear_gpu_memory)
    main()