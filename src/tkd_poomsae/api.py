"""Local API entry point."""

from fastapi import FastAPI

app = FastAPI(title="TKD Poomsae", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the empty local service is running."""
    return {"status": "ok"}
