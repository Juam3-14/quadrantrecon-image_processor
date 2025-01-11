from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.services.image_utils import detect_frame, process_image
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/images", tags=["Images"])

# Carpeta donde se almacenan temporalmente las imágenes cargadas
UPLOAD_DIR = Path("app/static")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/process")
async def upload_image(file: UploadFile = File(...)):
    """
    Cargar una imagen y procesarla para identificar y recortar el marco amarillo.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Solo se permiten imágenes JPEG o PNG.")
    
    # Guardar la imagen temporalmente
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Procesar la imagen
    try:
        processed_image_path = process_image(file_path)
        return JSONResponse(content={"processed_image": str(processed_image_path)}, status_code=200)
    except Exception as e:
        try:
            processed_image_path = detect_frame(file_path)
            return JSONResponse(content={"processed_image": str(processed_image_path)}, status_code=200)
        except:
            raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")
