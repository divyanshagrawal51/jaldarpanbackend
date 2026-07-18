import os
import json
import hashlib
import base64
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.5-flash")

# ── GROQ FALLBACK CLIENT ──
# Text tasks (analyze-from-text/farmer/swaps) use Groq as PRIMARY — Gemini
# is only touched if Groq fails. This keeps Gemini's tight daily quota
# reserved for image tasks (scan / analyze-from-image), which have no
# solid fallback since Groq's vision model is preview/unreliable.
GROQ_TEXT_MODEL = "openai/gpt-oss-20b"
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"  # preview model — may change without notice

try:
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
except Exception:
    groq_client = None


def _log(task: str, provider: str):
    tags = {
        "gemini": "🟦 GEMINI",
        "groq": "🟩 GROQ (fallback)",
        "cache": "⬜ CACHE (no API call)",
        "none": "🟥 FAILED (no provider worked)",
    }
    print(f"[{task}] {tags.get(provider, provider)}")

# ── SIMPLE IN-MEMORY CACHE ──
# If the same request (same image / same items) comes in again during
# testing or a live demo, don't spend quota re-asking the model.
_cache: dict = {}


def _cache_get(key: str):
    return _cache.get(key)


def _cache_set(key: str, value):
    _cache[key] = value
    return value


def _hash_image(image_base64: str) -> str:
    return hashlib.sha256(image_base64.encode("utf-8")).hexdigest()


def _call_groq_text(prompt: str) -> str | None:
    if not groq_client:
        return None
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Groq text fallback error: {e}")
        return None


def _call_groq_vision(image_base64: str, prompt: str) -> str | None:
    if not groq_client:
        return None
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }},
                ],
            }],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Groq vision fallback error: {e}")
        return None


def _parse_gemini_json(text: str) -> dict | None:
    try:
        clean = text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        print(f"JSON parse error: {e}\nRaw: {text}")
        return None


# ── FOOD IDENTIFICATION FROM IMAGE (/scan) ──

IDENTIFY_PROMPT = (
    "Identify the main food or agricultural product in this image. "
    "Reply with ONLY the food name, one word or two words max. "
    "Examples: rice, tomato, beef, green pepper. "
    "If no food is visible, reply with: none"
)


def identify_food_from_image(image_base64: str) -> str | None:
    cache_key = f"identify:{_hash_image(image_base64)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        _log("identify_food_from_image", "cache")
        return cached if cached != "__none__" else None

    image_data = base64.b64decode(image_base64)
    food_name = None

    try:
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": image_data},
            IDENTIFY_PROMPT
        ])
        food_name = response.text.strip().lower()
        _log("identify_food_from_image", "gemini")
    except Exception as e:
        print(f"Gemini error (identify_food_from_image): {e}")
        food_name = None

    # Gemini unreachable/errored -> try Groq vision
    if food_name is None:
        groq_text = _call_groq_vision(image_base64, IDENTIFY_PROMPT)
        if groq_text:
            food_name = groq_text.strip().lower()
            _log("identify_food_from_image", "groq")

    if not food_name or food_name == "none":
        if food_name is None:
            _log("identify_food_from_image", "none")
        _cache_set(cache_key, "__none__")
        return None

    _cache_set(cache_key, food_name)
    return food_name


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


def analyze_meal_from_image(image_base64: str) -> dict | None:
    cache_key = f"analyze_img:{_hash_image(image_base64)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        _log("analyze_meal_from_image", "cache")
        return cached

    prompt_text = (
        "First identify all the food items visible in this meal image. "
        "Then estimate the water footprint of the complete meal.\n\n"
        + ANALYSIS_INSTRUCTIONS
    )
    image_data = base64.b64decode(image_base64)
    result = None

    try:
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": image_data},
            prompt_text
        ])
        result = _parse_gemini_json(response.text)
        if result:
            _log("analyze_meal_from_image", "gemini")
    except Exception as e:
        print(f"Gemini error (analyze_meal_from_image): {e}")
        result = None

    if result is None:
        groq_text = _call_groq_vision(image_base64, prompt_text)
        if groq_text:
            result = _parse_gemini_json(groq_text)
            if result:
                _log("analyze_meal_from_image", "groq")

    if result is None:
        _log("analyze_meal_from_image", "none")

    if result:
        _cache_set(cache_key, result)
    return result


def analyze_meal_from_text(items: list) -> dict | None:
    meal_description = "\n".join(
        f"- {item.name}: {item.quantity}" for item in items
    )
    cache_key = f"analyze_text:{meal_description}"
    cached = _cache_get(cache_key)
    if cached is not None:
        _log("analyze_meal_from_text", "cache")
        return cached

    prompt = f"Meal contents:\n{meal_description}\n\n" + ANALYSIS_INSTRUCTIONS
    result = None

    # Groq first — this is a text-only task, so it doesn't need to touch
    # Gemini's limited daily quota at all. Gemini quota stays reserved
    # for image tasks (/scan, analyze-from-image) which have no solid fallback.
    groq_text = _call_groq_text(prompt)
    if groq_text:
        result = _parse_gemini_json(groq_text)
        if result:
            _log("analyze_meal_from_text", "groq")

    if result is None:
        try:
            response = model.generate_content(prompt)
            result = _parse_gemini_json(response.text)
            if result:
                _log("analyze_meal_from_text", "gemini")
        except Exception as e:
            print(f"Gemini error (analyze_meal_from_text): {e}")
            result = None

    if result is None:
        _log("analyze_meal_from_text", "none")

    if result:
        _cache_set(cache_key, result)
    return result


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

CRITICAL RULE — swaps must stay usable in the SAME dish:
Every item in this meal is playing a specific role (e.g. the bread in a sandwich,
the protein in a curry, the base grain in a bowl). A swap is only valid if the
replacement can physically take that same role and the dish still makes sense
as the same kind of meal after swapping.
- A bread/bun/roti used to hold or wrap other ingredients must be swapped for
  another bread-like item that can do the same job (e.g. whole-wheat bread,
  multigrain bread, a thinner roti/wrap) — never for something that isn't
  bread-like, even if it has a lower footprint (e.g. do NOT suggest swapping
  sandwich bread for plain roti/rice — you cannot make a sandwich with those).
- A meat/protein filling must be swapped for another filling-appropriate protein
  that fits the same preparation (e.g. paneer, tofu, a lentil/rajma patty,
  soy chunks) — never for something served in an unrelated form
  (e.g. do NOT suggest swapping chicken in a sandwich for plain dal/lentil curry
  — that cannot go inside a sandwich).
- If you cannot think of a same-role, same-format alternative for an item,
  skip that item rather than suggesting an incompatible one.

Only suggest swaps where the water saving is at least 50 litres.
Alternatives must be realistic, commonly available Indian foods.
If all items are already low water footprint, or no valid same-format swap
exists, return an empty array [].
Reply with ONLY a valid JSON array matching this exact schema, no markdown, no explanation:
""" + SWAP_SCHEMA


def get_swap_suggestions(items: list) -> list:
    """
    items: list of dicts with keys: name, quantity, litres
    Returns a list of swap suggestion dicts.
    """
    item_lines = "\n".join(
        f"- {item['name']} ({item['quantity']}): {item['litres']}L"
        for item in items
    )
    cache_key = f"swaps:{item_lines}"
    cached = _cache_get(cache_key)
    if cached is not None:
        _log("get_swap_suggestions", "cache")
        return cached

    prompt = f"Meal items with water footprint:\n{item_lines}\n\n" + SWAP_INSTRUCTIONS
    result = None

    # Groq first — text-only task, keep Gemini quota for image tasks.
    groq_text = _call_groq_text(prompt)
    if groq_text:
        parsed = _parse_gemini_json(groq_text)
        if isinstance(parsed, list):
            result = parsed
            _log("get_swap_suggestions", "groq")

    if not isinstance(result, list):
        try:
            response = model.generate_content(prompt)
            parsed = _parse_gemini_json(response.text)
            if isinstance(parsed, list):
                result = parsed
                _log("get_swap_suggestions", "gemini")
        except Exception as e:
            print(f"Gemini error (get_swap_suggestions): {e}")

    if not isinstance(result, list):
        _log("get_swap_suggestions", "none")
        result = []

    _cache_set(cache_key, result)
    return result


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
    cache_key = f"farm:{crop}:{area}:{irrigation}:{region}:{soil}:{water_source}"
    cached = _cache_get(cache_key)
    if cached is not None:
        _log("analyze_farm", "cache")
        return cached

    result = None

    # Groq first — text-only task, keep Gemini quota for image tasks.
    groq_text = _call_groq_text(prompt)
    if groq_text:
        result = _parse_gemini_json(groq_text)
        if result:
            _log("analyze_farm", "groq")

    if result is None:
        try:
            response = model.generate_content(prompt)
            result = _parse_gemini_json(response.text)
            if result:
                _log("analyze_farm", "gemini")
        except Exception as e:
            print(f"Gemini error (analyze_farm): {e}")

    if result is None:
        _log("analyze_farm", "none")

    if result:
        result['total_litres'] = int(result.get('total_litres', 0))
        result['efficiency'] = int(result.get('efficiency', 0))
        result['saving_potential'] = int(result.get('saving_potential', 0))
        _cache_set(cache_key, result)

    return result
