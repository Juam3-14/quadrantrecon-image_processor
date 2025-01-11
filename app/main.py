from fastapi import FastAPI
from app.routers import images

app = FastAPI(title="Image Processing API")

# Incluir rutas
app.include_router(images.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Image Processing API"}
