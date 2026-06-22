from fastapi import APIRouter
from pydantic import BaseModel
from services.gemini_service import analyze_farm

router = APIRouter()

class FarmRequest(BaseModel):
    crop: str
    area: float
    irrigation: str
    region: str = ""
    soil: str = ""
    water_source: str = ""

@router.post("/farmer")
def farmer(request: FarmRequest):
    result = analyze_farm(
        crop=request.crop,
        area=request.area,
        irrigation=request.irrigation,
        region=request.region,
        soil=request.soil,
        water_source=request.water_source
    )

    if result is None:
        return {"success": False, "message": "Gemini could not analyze this farm."}

    return {"success": True, **result}
