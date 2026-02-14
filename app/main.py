from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes.summarize import router as summarize_router
from app.db.base import Base
from app.db.session import engine
from app.core.errors import APIError
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI(title="AI Service Core")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables
Base.metadata.create_all(bind=engine)

# Routes
app.include_router(summarize_router, prefix="/api")

@app.get("/")
def health():
    return {"status": "running"}


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(status_code=exc.status_code,content={"error": exc.message} )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400,content={"error": "Bad request","message": str(exc)},
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    error = exc.errors()[0]

    field = error["loc"][-1]
    error_type = error["type"]

    if error_type == "missing":
        message = f"{field.replace('_', ' ').title()} is required"

    elif error_type == "int_parsing":
        message = f"{field.replace('_', ' ').title()} must be a number"

    elif error_type == "string_too_short":
        message = f"{field.replace('_', ' ').title()} is too short"

    else:
        message = f"Invalid value for {field.replace('_', ' ')}"

    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad request",
            "message": message,
        },
    )

@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# uvicorn main:app --reload