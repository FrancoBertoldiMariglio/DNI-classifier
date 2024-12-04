from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import create_router
from api.routes import lifespan

def create_application() -> FastAPI:
    app = FastAPI(
        title="RelationNet API",
        lifespan=lifespan
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize and include routes with dependency injection
    router = create_router()
    app.include_router(router, prefix="/api/v1")

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)