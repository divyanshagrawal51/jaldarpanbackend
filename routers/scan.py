from fastapi import APIRouter
from pydantic import BaseModel
from services.gemini_service import identify_food_from_image
from services.db_service import get_footprint

router = APIRouter()

class ScanRequest(BaseModel):
    image: str  # base64 string

@router.post("/scan")
def scan(request: ScanRequest):
    print("SCAN ENDPOINT HIT")
    food_name = identify_food_from_image(request.image)

    print("Gemini returned:", food_name)

    if food_name is None:
        return {
            "found": False,
            "message": "Could not identify any food in the image"
        }

    result = get_footprint(food_name)

    print("DB result:", result)
    
    if result is None:
        return {
            "found": False,
            "identified_as": food_name,
            "message": f"Identified '{food_name}' but no water data found for it"
        }
    
    return {
        "found": True,
        "identified_as": food_name,
        **result
    }