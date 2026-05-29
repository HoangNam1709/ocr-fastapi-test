from fastapi import FastAPI, UploadFile, File
import shutil
import os
from app.ocr_service import extract_cccd

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def root():
    return {"message": "OCR API running"}


@app.post("/ocr/cccd")
async def ocr_cccd(image: UploadFile = File(...)):
    # Lưu file tạm
    file_path = os.path.join(UPLOAD_FOLDER, image.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Gọi OCR
    result = extract_cccd(file_path)

    return {
        "status": "SUCCESS",
        "data": result
    }