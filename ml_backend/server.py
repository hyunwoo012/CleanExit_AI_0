from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from ultralytics import YOLO
import tempfile
import os
import requests
import cv2

app = FastAPI()

# ----------------------
# Health & Setup
# ----------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/setup")
async def setup():
    return {
        "model_name": "CleanExit YOLOv8",
        "model_version": "1.0",
        "description": "YOLOv8 backend for ExitGuard",
    }


# ----------------------
# CORS
# ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Load YOLO model
# ----------------------
model = YOLO("yolov8n.pt")  # 프리트레인 모델 사용


# ----------------------
# Label mappings
# ----------------------
CLASS_MAP = {
    0: "trash",
    1: "book",
    2: "misc-object",
}


# ----------------------
# LS request schema
# ----------------------
class LSRequest(BaseModel):
    tasks: list
    project: str | None = None
    label_config: str | None = None
    params: dict | None = None


# ----------------------
# Prediction Endpoint
# ----------------------
@app.post("/predict")
async def predict(request: LSRequest):

    print("🔥 Incoming body:", request.model_dump())

    # 1) Label Studio가 보내는 첫 번째 task의 이미지 경로 가져오기
    try:
        task = request.tasks[0]
        raw_url = task["data"]["image"]
    except:
        return JSONResponse({"detail": "Invalid request format"}, status_code=400)

    # 2) URL 변환: /data/... → http://localhost:8080/data/...
    if raw_url.startswith("/"):
        image_url = f"http://localhost:8080{raw_url}"
    else:
        image_url = raw_url

    print("📌 Final image URL:", image_url)

    # 3) 이미지 다운로드
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        r = requests.get(image_url, timeout=5)
        tmp.write(r.content)
        tmp.close()
    except Exception as e:
        return JSONResponse({"detail": f"Image download failed: {str(e)}"}, status_code=500)

    print("📌 Downloaded file size:", os.path.getsize(tmp.name))

    # 4) CV2 이미지 읽기 실패 체크
    img = cv2.imread(tmp.name)
    if img is None:
        print("⚠️ CV2 failed to read image:", tmp.name)
        return JSONResponse(
            {"detail": "Image decode failed (OpenCV could not read the file)"}, status_code=500
        )

    # 5) YOLO inference
    try:
        result = model(tmp.name)[0]
    except Exception as e:
        return JSONResponse({"detail": f"YOLO inference failed: {str(e)}"}, status_code=500)

    predictions = []

    # 6) 박스 생성
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])

        ls_label = CLASS_MAP.get(cls_id, "misc-object")

        predictions.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "rectanglelabels": [ls_label],
            },
        })

    return [{
        "result": predictions,
        "score": 0.9
    }]


# ----------------------
# Run server
# ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
