import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import base64

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def identify_food_from_image(image_base64: str) -> str | None:
    try:
        image_data = base64.b64decode(image_base64)

        response = model.generate_content([
            {
                "mime_type": "image/jpeg",
                "data": image_data
            },
            "Identify the main food or agricultural product in this image. "
            "Reply with ONLY the food name, one word or two words max. "
            "Examples: rice, tomato, beef, green pepper. "
            "If no food is visible, reply with: none"
        ])

        food_name = response.text.strip().lower()

        if food_name == "none":
            return None

        return food_name

    except Exception as e:
        print(f"Gemini error: {e}")
        return None


ANALYSIS_SCHEMA = """
{
  "total_litres": <number, total water footprint of the whole meal>,
  "green": <number, green water component>,
  "blue": <number, blue water component>,
  "grey": <number, grey water component>,
  "items": [
    {
      "name": "<food item name>",
      "quantity": "<quantity as described>",
      "litres": <number, water footprint for this item at this quantity>
    }
  ],
  "summary": "<one sentence insight about this meal's water footprint>"
}
"""

ANALYSIS_INSTRUCTIONS = """
You are a water footprint expert. Estimate the water footprint of the given meal.
Use standard water footprint research values (Hoekstra et al.) as reference.
All values are in LITRES.
Green water = rainwater consumed by crops.
Blue water = surface/groundwater consumed.
Grey water = freshwater needed to dilute pollutants.
total_litres must equal green + blue + grey.
Quantities should be interpreted as realistic Indian serving sizes.
Reply with ONLY a valid JSON object matching this exact schema, no markdown, no explanation:
""" + ANALYSIS_SCHEMA


def _parse_gemini_json(text: str) -> dict | None:
    try:
        clean = text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        print(f"JSON parse error: {e}\nRaw: {text}")
        return None


def analyze_meal_from_image(image_base64: str) -> dict | None:
    try:
        image_data = base64.b64decode(image_base64)

        response = model.generate_content([
            {
                "mime_type": "image/jpeg",
                "data": image_data
            },
            "First identify all the food items visible in this meal image. "
            "Then estimate the water footprint of the complete meal.\n\n"
            + ANALYSIS_INSTRUCTIONS
        ])

        return _parse_gemini_json(response.text)

    except Exception as e:
        print(f"Gemini analyze_from_image error: {e}")
        return None

def analyze_meal_from_text(items: list) -> dict | None:
    try:
        meal_description = "\n".join(
            f"- {item.name}: {item.quantity}" for item in items
        )

        prompt = (
            f"Meal contents:\n{meal_description}\n\n"
            + ANALYSIS_INSTRUCTIONS
        )

        response = model.generate_content(prompt)
        print(f"Gemini raw response: {response.text}")  # ADD THIS
        return _parse_gemini_json(response.text)

    except Exception as e:
        print(f"Gemini analyze_from_text error: {e}")
        return None
