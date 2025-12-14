import cv2
import os
from pathlib import Path

video_folder = "origin_video"   # 영상 폴더
output_folder = "frames_v2"     # 프레임 저장 폴더
os.makedirs(output_folder, exist_ok=True)

FRAME_INTERVAL = 2              # 5프레임마다 1장 저장
global_idx = 0                 # 전체 이미지 번호

video_files = list(Path(video_folder).glob("*.MOV"))

for video_path in video_files:
    print(f"Processing {video_path.name} ...")
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # N프레임마다 1장 저장
        if frame_idx % FRAME_INTERVAL == 0:
            # 무조건 90도 회전
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            save_path = os.path.join(output_folder, f"frame_{global_idx:06d}.jpg")
            cv2.imwrite(save_path, frame)
            global_idx += 1

        frame_idx += 1

    cap.release()

print("모든 영상 프레임 추출 완료!")
print(f"총 저장된 이미지 수: {global_idx}")
