from fastapi import APIRouter
from pydantic import BaseModel
from services.db_service import get_footprint

router = APIRouter()

class LookupRequest(BaseModel):
    food_name: str

@router.post("/lookup")
def lookup(request: LookupRequest):
    result = get_footprint(request.food_name)
    
    if result is None:
        return {
            "found": False,
            "message": f"No data found for '{request.food_name}'"
        }
    
    return {
        "found": True,
        **result
    }