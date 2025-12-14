import json
from pathlib import Path
from ultralytics import YOLO

model = YOLO("runs_cleanexit/yolov8n_cleanexit_v1/weights/best.pt")
image_folder = Path("frames_v2")

tasks = []

def to_labelstudio_path(local_path: Path):
    # Label Studio가 사용하는 URL 형태로 변환
    return f"/data/local-files/?d={local_path.as_posix()}"

for img_path in image_folder.glob("*.jpg"):
    results = model(img_path)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = r.orig_shape[1]
            height = r.orig_shape[0]

            detections.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "score": conf,
                "value": {
                    "rectanglelabels": [model.names[cls]],
                    "x": x1 / width * 100,
                    "y": y1 / height * 100,
                    "width": (x2 - x1) / width * 100,
                    "height": (y2 - y1) / height * 100
                }
            })

    # 여기서 경로만 제대로 맞추면 됨!!!
    tasks.append({
        "data": {"image": to_labelstudio_path(img_path)},
        "predictions": [{
            "model_version": "yolo_pretrain_100",
            "result": detections
        }]
    })

with open("auto_labels.json", "w") as f:
    json.dump(tasks, f, indent=2)

print("라벨 스튜디오용 자동 라벨링 JSON 생성 완료 → auto_labels.json")
