import io
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
from PIL import Image
import os

app = FastAPI()
model = YOLO("/home/saybrone/FLL-AI/FLLAI.pt")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>FLL AI</h2>
    <form action="/result" enctype="multipart/form-data" method="post">
        <input name="file" type="file" accept="image/*">
        <button type="submit">Detect</button>
    </form>
    """
@app.post("/result")
async def test(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")


    results = model(image)[0]
    result_img = results.plot()
    a, encoded = cv2.imencode(".jpg", result_img)
    if a != True:
        return
    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg"
    )
#uvicorn server:app --reload --host 0.0.0.0 --port 8000