import os
import cv2
import json
import pathlib
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from insightface.app import FaceAnalysis
import requests

# Create the output directory if it doesn't exist
output_dir = "json_maps"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

# Function to resolve relative paths
def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

# Generalized function to detect the dominant color in a region
def detect_hex_color(region_img):
    if region_img is None or region_img.size == 0:
        return '#000000'  # Return black as a fallback color

    region_img = cv2.resize(region_img, (50, 25))
    region_img = region_img.reshape((region_img.shape[0] * region_img.shape[1], 3))

    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(region_img)
    label_counts = Counter(labels)

    dominant_color = kmeans.cluster_centers_[label_counts.most_common(1)[0][0]]
    dominant_color_rgb = dominant_color[::-1]
    dominant_color_hex = '#%02x%02x%02x' % (int(dominant_color_rgb[0]), int(dominant_color_rgb[1]), int(dominant_color_rgb[2]))
    
    return dominant_color_hex

def get_color_name_from_api(hex_color):
    response = requests.get(f"https://www.thecolorapi.com/id?hex={hex_color.lstrip('#')}")
    if response.status_code == 200:
        color_data = response.json()
        return color_data['name']['value']
    return "unknown"

def get_eye_regions(image, landmarks):
    left_eye_coords = landmarks["eyes"]["left_eye"]
    right_eye_coords = landmarks["eyes"]["right_eye"]

    left_eye_region = image[max(0, int(left_eye_coords[1]) - 10):min(image.shape[0], int(left_eye_coords[1]) + 10),
                            max(0, int(left_eye_coords[0]) - 10):min(image.shape[1], int(left_eye_coords[0]) + 10)]
    right_eye_region = image[max(0, int(right_eye_coords[1]) - 10):min(image.shape[0], int(right_eye_coords[1]) + 10),
                             max(0, int(right_eye_coords[0]) - 10):min(image.shape[1], int(right_eye_coords[0]) + 10)]

    return left_eye_region, right_eye_region

def get_facial_hair_region(image, landmarks):
    mouth_left_corner = landmarks["mouth"]["left_corner"]
    mouth_right_corner = landmarks["mouth"]["right_corner"]

    facial_hair_region = image[max(0, int(mouth_left_corner[1]) - 20):min(image.shape[0], int(mouth_left_corner[1]) + 20),
                               max(0, int(mouth_left_corner[0]) - 20):min(image.shape[1], int(mouth_right_corner[0]) + 20)]

    return facial_hair_region

def get_head_hair_region(image, landmarks):
    eye_left = int(landmarks["eyes"]["left_eye"][0])
    eye_right = int(landmarks["eyes"]["right_eye"][0])
    nose_bottom = int(landmarks["nose"][1])
    
    hair_start = max(0, nose_bottom - 60)
    hair_end = nose_bottom - 20
    hair_region = image[hair_start:hair_end, eye_left:eye_right]
    
    return hair_region

def generate_face_profile(image_path: str) -> dict:
    app = FaceAnalysis()
    app.prepare(ctx_id=0)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image at {image_path}")

    faces = app.get(img)
    if not faces:
        raise ValueError("No face detected in the image.")

    face = faces[0]
    embedding = face.embedding.tolist()

    landmarks = {
        "eyes": {
            "left_eye": face.landmark_2d_106[36].tolist(),
            "right_eye": face.landmark_2d_106[45].tolist()
        },
        "nose": face.landmark_2d_106[30].tolist(),
        "mouth": {
            "left_corner": face.landmark_2d_106[48].tolist(),
            "right_corner": face.landmark_2d_106[54].tolist()
        }
    }

    left_eye_region, right_eye_region = get_eye_regions(img, landmarks)

    # Validate if the eye regions are non-empty before detection
    if left_eye_region.size > 0:
        left_eye_color = detect_hex_color(left_eye_region)
        left_eye_color_guess = get_color_name_from_api(left_eye_color)
    else:
        left_eye_color = "#000000"
        left_eye_color_guess = "unknown"

    if right_eye_region.size > 0:
        right_eye_color = detect_hex_color(right_eye_region)
        right_eye_color_guess = get_color_name_from_api(right_eye_color)
    else:
        right_eye_color = "#000000"
        right_eye_color_guess = "unknown"

    facial_hair_region = get_facial_hair_region(img, landmarks)
    facial_hair_color = detect_hex_color(facial_hair_region)
    facial_hair_color_name = get_color_name_from_api(facial_hair_color)

    head_hair_region = get_head_hair_region(img, landmarks)
    head_hair_color = detect_hex_color(head_hair_region)
    head_hair_color_name = get_color_name_from_api(head_hair_color)

    return {
        "reference_images": [
            {"pose": "front", "embedding": embedding}
        ],
        "facial_landmarks": landmarks,
        "physical_features": {
            "left_eye_color": left_eye_color,
            "right_eye_color": right_eye_color,
            "left_eye_color_guess": left_eye_color_guess,
            "right_eye_color_guess": right_eye_color_guess,
            "facial_hair": {
                "type": "unknown",
                "color": facial_hair_color,
                "color_guess": facial_hair_color_name,
            },
            "head_hair": {
                "color": head_hair_color,
                "color_guess": head_hair_color_name,
            },
            "skin_tone": "unknown"
        },
        "accessories": {
            "hat": {"type": "unknown", "color": "unknown"},
            "glasses": {"type": "unknown", "color": "unknown"},
            "earrings": "unknown"
        },
        "clothing": {
            "upper_body": {"type": "unknown", "color": "unknown", "pattern": "unknown"},
            "lower_body": {"type": "unknown", "color": "unknown"}
        },
        "contextual_features": {
            "height": "unknown",
            "build": "unknown",
            "movement": "unknown"
        }
    }