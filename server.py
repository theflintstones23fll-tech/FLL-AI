import io
import cv2
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
from PIL import Image
import os
def calculate_len_xy(box_xyxy,len_coef):
    lenx = (box_xyxy[2] - box_xyxy[0])*len_coef
    leny = (box_xyxy[3] - box_xyxy[1])*len_coef
    return[lenx,leny]

def print_results(results):
    boxes = results.boxes
    classes = boxes.cls.tolist()
    for clas in classes:
        if clas == 1:
            index_of_meter = classes.index(clas)
    xyxy = boxes.xyxy.tolist()
    boxes_of_meter = xyxy[index_of_meter]
    len_x_of_meter = boxes_of_meter[2]-boxes_of_meter[0]
    len_coef = 8/len_x_of_meter
    response = []
    rtn = ""
    for box_xyxy in xyxy:
        if xyxy.index(box_xyxy) != index_of_meter:
            response.append(calculate_len_xy(box_xyxy,len_coef))
    for i in response:
        rtn = rtn + f"Index: {response.index(i)} ----> Length_x: {i[0]}, Length_y: {i[1]} <br/> <br/>"
    return rtn




app = FastAPI()
model = YOLO("FLL-AI.pt")
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>FLL AI</h2>
    <form action="/detect" enctype="multipart/form-data" method="post">
        <input name="file" type="file" accept="image/*" required><br>
        <button type="submit">Detect</button>
    </form>
    """



@app.post("/detect", response_class=HTMLResponse)
async def test(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model.predict(image, conf=0.65)[0]
    result_img = results.plot()
    a, encoded = cv2.imencode(".jpg", result_img)
    if not a:
        return "Error"
    foto = base64.b64encode(encoded).decode()
    return f"""
        <body style="max-width: 800px; margin: auto;">
            <h2>Test Result</h2>
            <img src="data:image/jpeg;base64,{foto}" 
                 style="max-width: 100%;"/>
            <h3>Details</h3>
            <p style="font-size = font-family: monospace;">
                {print_results(results)}
            </p>
            <a href="/">Back</a>
        </body>
    """

    
#uvicorn server:app --reload --host 0.0.0.0 --port 8000
