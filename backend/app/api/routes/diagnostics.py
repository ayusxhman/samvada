from fastapi import APIRouter

from backend.app.orchestration import ai_orchestrator

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


@router.get("")
async def get_diagnostics():
    """Expose measured operation latency and real model readiness."""
    return ai_orchestrator.diagnostics()
