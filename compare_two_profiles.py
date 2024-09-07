import os
import json
from difflib import SequenceMatcher

def load_profile(path):
    with open(path, 'r') as file:
        return json.load(file)

def compare_strings(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

def compare_numbers(num1, num2, tolerance=5):
    return 1 - min(abs(num1 - num2) / tolerance, 1)

def compare_colors(color1, color2):
    return compare_strings(color1.lower(), color2.lower())

def compare_landmarks(landmarks1, landmarks2):
    if len(landmarks1) != len(landmarks2):
        return 0
    similarity = 0
    for p1, p2 in zip(landmarks1, landmarks2):
        similarity += compare_numbers(p1, p2)
    return similarity / len(landmarks1)

def compare_feature(value1, value2):
    if value1 is None or value2 is None:
        return 0 if value1 != value2 else 1  # Handles None vs None as 100% similar
    
    if isinstance(value1, str):
        return compare_strings(value1, value2)
    elif isinstance(value1, (int, float)):
        return compare_numbers(value1, value2)
    elif isinstance(value1, list):
        return compare_landmarks(value1, value2)
    
    return 0  # default case for unknown types

def calculate_similarity(profile1, profile2):
    features = [
        ("body_structure.pose", 0.1),
        ("head.physical_features.eyes.color", 0.1),
        ("head.physical_features.facial_hair.llava13b_guess", 0.05),
        ("head.physical_features.head_hair.llava13b_guess", 0.05),
        ("head.gender.value", 0.1),
        ("head.physical_features.skin_tone.llava13b_guess", 0.1),
        ("head.wearing_hat.present", 0.1),
        ("accessories.glasses.present", 0.05),
        ("clothing.upper_body.type", 0.05),
        ("clothing.upper_body.color", 0.05),
        ("clothing.lower_body.type", 0.05),
        ("clothing.lower_body.color", 0.05),
        ("description", 0.05),
        ("age_estimation.value", 0.05),
        ("reference_images[0].embedding", 0.1)  # Special handling for embedding comparison
    ]

    total_weight = sum(weight for key, weight in features)
    similarity_sum = 0

    for feature, weight in features:
        keys = feature.split('.')
        val1 = profile1
        val2 = profile2

        try:
            for key in keys:
                if key.endswith("]"):
                    base, idx = key[:-1].split("[")
                    idx = int(idx)
                    val1 = val1[base][idx]
                    val2 = val2[base][idx]
                else:
                    val1 = val1[key]
                    val2 = val2[key]

            similarity = compare_feature(val1, val2)
            similarity_sum += similarity * weight

            # Debugging statements
            print(f"Feature: {feature}")
            print(f"Value 1: {val1}")
            print(f"Value 2: {val2}")
            print(f"Similarity: {similarity}")
            print(f"Weighted Similarity: {similarity * weight}")
            print()

        except (KeyError, IndexError) as e:
            print(f"Feature {feature} caused an error: {e}")
            similarity_sum += 0  # If the feature is missing in one of the profiles

    return (similarity_sum / total_weight) * 100

def main():
    # Paths to JSON files
    bob_profile_path = "json_profiles/cat_png.json"
    andy_profile_path = "json_profiles/andy_jpg.json"

    # Load profiles
    bob_profile = load_profile(bob_profile_path)
    andy_profile = load_profile(andy_profile_path)

    # Calculate similarity
    similarity_score = calculate_similarity(bob_profile, andy_profile)
    print(f"Cat and Andy's profiles are {similarity_score:.5f}% similar.")

if __name__ == "__main__":
    main()