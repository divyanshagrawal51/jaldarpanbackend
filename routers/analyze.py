from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from services.gemini_service import analyze_meal_from_image, analyze_meal_from_text

router = APIRouter()

class MealItem(BaseModel):
    name: str
    quantity: str  # e.g. "2 chapatis", "1 bowl", "200g"

class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    items: Optional[List[MealItem]] = None

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    if request.image_base64:
        result = analyze_meal_from_image(request.image_base64)
    elif request.items:
        result = analyze_meal_from_text(request.items)
    else:
        return {"success": False, "message": "Provide either image_base64 or items."}

    if result is None:
        return {"success": False, "message": "Gemini could not analyze this meal."}

    return {"success": True, **result}
