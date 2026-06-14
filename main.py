import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import lookup, scan, analyze

app = FastAPI()
app.include_router(lookup.router)
app.include_router(scan.router)
app.include_router(analyze.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


@app.get("/")
def root():
    return {"status": "running"}
