from fastapi import APIRouter
from backend.app.config import SUPPORTED_LANGUAGES

router = APIRouter(prefix="/languages", tags=["Languages"])

@router.get("")
@router.get("/")
async def get_languages():
    """
    Returns a list of all supported languages with their codes and names.
    """
    # Convert dict to sorted list of objects
    langs = [{"code": code, "name": name} for code, name in SUPPORTED_LANGUAGES.items()]
    # Sort alphabetically by language name
    langs.sort(key=lambda x: x["name"])
    return {"languages": langs}
