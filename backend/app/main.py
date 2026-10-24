import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.config import BASE_DIR, DATASET_DIR, FRONTEND_ORIGINS
from backend.app.api.routes import avatar, diagnostics, languages, recognize, translate
from backend.app.api import websocket
from backend.app.services.sign_to_text import sign_to_text_service

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Sign Language Translation Bot API",
    description="Backend services for fingerspelling, word sign recognition, text/speech-to-sign translations, and VRM avatar animations.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.app.api.routes import translate, recognize, avatar, languages, diagnostics, recorder
from backend.app.api import websocket

# Include API Routers
app.include_router(translate.router, prefix="/api")
app.include_router(recognize.router, prefix="/api")
app.include_router(avatar.router, prefix="/api")
app.include_router(languages.router, prefix="/api")
app.include_router(diagnostics.router, prefix="/api")
app.include_router(recorder.router)
app.include_router(websocket.router, prefix="/api")

# Serve dataset folder as static files
# This allows frontend to fetch videos from dataset/wlasl/videos/ and animations from dataset/animation_clips/
if DATASET_DIR.exists():
    app.mount("/dataset", StaticFiles(directory=str(DATASET_DIR)), name="dataset")
    logger.info(f"Mounted static files at /dataset from {DATASET_DIR}")
else:
    logger.warning(f"Dataset directory not found at {DATASET_DIR}. Static serving disabled.")

# Serve VRM file directly
vrm_path = BASE_DIR / "eric1.vrm"
if vrm_path.exists():
    app.mount("/vrm", StaticFiles(directory=str(BASE_DIR)), name="vrm")
    logger.info(f"Mounted /vrm to serve {vrm_path.name}")

@app.get("/health/ready")
async def readiness():
    """Report which real inference capabilities are currently available."""
    models = sign_to_text_service.readiness()
    return {
        "ready": models["fingerspelling"]["available"] or models["word_recognition"]["available"],
        "models": models,
    }

# Mount Compiled Next.js Static UI at Root (Option 3 Single-Server Execution)
frontend_out = BASE_DIR / "frontend" / "out"
if frontend_out.exists():
    app.mount("/", StaticFiles(directory=str(frontend_out), html=True), name="frontend")
    logger.info(f"Mounted Next.js Static UI at root / from {frontend_out}")
else:
    @app.get("/")
    async def root():
        return {
            "message": "AI Sign Language Translation Bot Backend running successfully.",
            "docs_url": "/docs",
            "frontend_status": "Static build not found. Run 'npm run build' in frontend directory to enable Option 3."
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
