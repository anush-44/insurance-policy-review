"""PolicyLens FastAPI application entry point.

Start with:
    uvicorn app.main:app --reload --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import get_pipeline
from app.api.analyse import router as analyse_router
from app.api.policies import router as policies_router
from app.api.qa import router as qa_router
from app.core.config import settings
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: warm up the RAG pipeline on startup."""
    logger.info(f"Starting {settings.app_name} — warming up RAG pipeline...")
    # Eagerly initialise the singleton so first request doesn't pay cold-start cost
    get_pipeline()
    logger.info("RAG pipeline ready.")
    yield
    logger.info(f"{settings.app_name} shutting down.")


app = FastAPI(
    title="PolicyLens API",
    description=(
        "Grounded Indian Health Insurance Policy Understanding. "
        "All answers are cited to exact clauses and pages."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(policies_router)
app.include_router(qa_router)
app.include_router(analyse_router)


@app.get("/", tags=["health"])
async def root() -> dict:
    """Health check endpoint."""
    return {
        "service": settings.app_name,
        "version": "0.1.0",
        "status": "healthy",
        "disclaimer": settings.disclaimer,
    }


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Liveness probe for Docker / Kubernetes."""
    return {"status": "ok"}
