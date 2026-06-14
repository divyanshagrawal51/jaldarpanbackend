import json
import os
from rapidfuzz import process, fuzz

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/water_footprint.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    WATER_DATA = json.load(f)

CROP_NAMES = list(WATER_DATA.keys())

def get_footprint(food_name: str):
    food_name = food_name.lower().strip()
    
    # Exact match
    if food_name in WATER_DATA:
        return {
            "matched_food": food_name,
            **WATER_DATA[food_name]
        }
    
    # Fuzzy match
    match, score, _ = process.extractOne(food_name, CROP_NAMES, scorer=fuzz.WRatio)
    
    if score >= 70:
        return {
            "matched_food": match,
            **WATER_DATA[match]
        }
    
    return None