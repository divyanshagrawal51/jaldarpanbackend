import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import base64

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.5-flash")


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
        return _parse_gemini_json(response.text)

    except Exception as e:
        print(f"Gemini analyze_from_text error: {e}")
        return None


SWAP_SCHEMA = """
[
  {
    "original": "<food item name from the meal>",
    "swap": "<suggested lower water-footprint Indian alternative>",
    "originalLitres": <number>,
    "swapLitres": <number>,
    "saving": <number, originalLitres minus swapLitres>,
    "reason": "<one short sentence explaining why this swap saves water, in simple language>"
  }
]
"""

SWAP_INSTRUCTIONS = """
You are a water footprint expert advising Indian users on sustainable food choices.
Given a meal with per-item water footprint data, suggest 2-3 smart ingredient swaps.
Only suggest swaps where the water saving is at least 50 litres.
Alternatives must be realistic, commonly available Indian foods.
If all items are already low water footprint, return an empty array [].
Reply with ONLY a valid JSON array matching this exact schema, no markdown, no explanation:
""" + SWAP_SCHEMA


def get_swap_suggestions(items: list) -> list:
    """
    items: list of dicts with keys: name, quantity, litres
    Returns a list of swap suggestion dicts.
    """
    try:
        item_lines = "\n".join(
            f"- {item['name']} ({item['quantity']}): {item['litres']}L"
            for item in items
        )

        prompt = (
            f"Meal items with water footprint:\n{item_lines}\n\n"
            + SWAP_INSTRUCTIONS
        )

        response = model.generate_content(prompt)
        result = _parse_gemini_json(response.text)

        if isinstance(result, list):
            return result
        return []

    except Exception as e:
        print(f"Gemini get_swap_suggestions error: {e}")
        return []


FARM_SCHEMA = """
{
  "total_litres": <integer, total water used by this farm for one crop season>,
  "efficiency": <integer, irrigation efficiency percentage 0-100>,
  "saving_potential": <integer, litres that could be saved with better practices>,
  "tips": [
    "<practical water conservation tip specific to this crop, irrigation method, and region>",
    "<another tip>",
    "<another tip>"
  ]
}
"""

FARM_INSTRUCTIONS = """
You are an agricultural water footprint expert specializing in Indian farming.
Estimate the seasonal water footprint for the given farm and provide practical conservation tips.
Base your estimates on standard Indian agricultural water usage research.
All water values are in LITRES for the full crop season.
Efficiency is a percentage (0-100) reflecting how well current irrigation matches crop needs.
Saving potential is how many litres could be saved by switching to better practices.
Tips must be specific to the crop, irrigation method, soil type, region, and water source provided.
Reply with ONLY valid JSON matching this exact schema, no markdown, no explanation:
""" + FARM_SCHEMA


def analyze_farm(crop: str, area: float, irrigation: str,
                 region: str = "", soil: str = "", water_source: str = "") -> dict | None:
    try:
        prompt = (
            f"Farm details:\n"
            f"- Crop: {crop}\n"
            f"- Area: {area} acres\n"
            f"- Irrigation method: {irrigation}\n"
            f"- Region: {region or 'India (unspecified)'}\n"
            f"- Soil type: {soil or 'unspecified'}\n"
            f"- Water source: {water_source or 'unspecified'}\n\n"
            + FARM_INSTRUCTIONS
        )

        response = model.generate_content(prompt)
        result = _parse_gemini_json(response.text)

        if result:
            result['total_litres'] = int(result.get('total_litres', 0))
            result['efficiency'] = int(result.get('efficiency', 0))
            result['saving_potential'] = int(result.get('saving_potential', 0))

        return result

    except Exception as e:
        print(f"Gemini analyze_farm error: {e}")
        return None
