from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import attach_websockets, create_router
from app.core.config import get_settings
from app.services.session_manager import SessionManager


settings = get_settings()
manager = SessionManager(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await manager.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin],
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(create_router(manager), prefix=settings.api_prefix)
attach_websockets(app, manager)
