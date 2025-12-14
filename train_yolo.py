from ultralytics import YOLO
from pathlib import Path

def main():
    # 저장 경로를 절대경로로 명확히 지정
    SAVE_ROOT = Path("/Users/johyeon-u/PycharmProjects/CleanExit_AI_0/yolo_runs")
    SAVE_ROOT.mkdir(exist_ok=True)

    model = YOLO("yolov8s.pt")

    model.train(
        data="yolo_data-v4/data.yaml",
        epochs=120,
        imgsz=640,
        batch=8,
        patience=20,

        # augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=3.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.9,
        mixup=0.1,

        # 저장 경로 설정
        project=str(SAVE_ROOT),        # 상위 저장 폴더
        name="yolov8s_cleanexit_v4",  # 하위 실험 폴더 이름
        exist_ok=True                 # 덮어쓰기 허용
    )

    print("학습 끝! 🎉")
    print(f"결과 폴더: {SAVE_ROOT}/yolov8s_cleanexit_v4")
    print(f"best.pt 위치: {SAVE_ROOT}/yolov8s_cleanexit_v4/weights/best.pt")

if __name__ == "__main__":
    main()
