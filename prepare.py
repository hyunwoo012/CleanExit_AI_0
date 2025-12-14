import os
import shutil
import random
from pathlib import Path

import yaml

# ========= 설정 =========
# Label Studio에서 "YOLO with images"로 export 한 폴더
SRC = Path("auto_labeling_data")

# YOLO 학습용 최종 데이터셋이 저장될 폴더
OUT = Path("yolo_data-v4")

# train/val 비율
VAL_RATIO = 0.2

# 랜덤 시드 (항상 같은 분할을 얻고 싶으면 고정)
RANDOM_SEED = 42

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def prepare_dataset():
    """project-export → yolo_data(train/val)로 변환"""
    images_dir = SRC / "images"
    labels_dir = SRC / "labels"
    classes_path = SRC / "classes.txt"

    assert images_dir.exists(), f"{images_dir} 가 존재하지 않습니다."
    assert labels_dir.exists(), f"{labels_dir} 가 존재하지 않습니다."
    assert classes_path.exists(), f"{classes_path} 가 존재하지 않습니다."

    # 출력 디렉터리 생성
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    # 이미지 리스트 수집
    imgs = [
        p.name for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

    if not imgs:
        raise RuntimeError("이미지 파일을 찾지 못했습니다.")

    random.seed(RANDOM_SEED)
    random.shuffle(imgs)

    val_n = max(1, int(len(imgs) * VAL_RATIO))
    print(f"총 이미지: {len(imgs)}장 -> train: {len(imgs)-val_n}, val: {val_n}")

    skipped = 0

    for i, img_name in enumerate(imgs):
        split = "val" if i < val_n else "train"
        base = os.path.splitext(img_name)[0]

        src_img = images_dir / img_name
        src_lbl = labels_dir / f"{base}.txt"

        if not src_lbl.exists():
            print(f"[WARN] 라벨 파일 없음, 스킵: {src_lbl}")
            skipped += 1
            continue

        dst_img = OUT / "images" / split / img_name
        dst_lbl = OUT / "labels" / split / f"{base}.txt"

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

    if skipped:
        print(f"[INFO] 라벨이 없어 스킵한 이미지: {skipped}장")

    # classes.txt → data.yaml의 names 생성
    names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    names_dict = {i: name for i, name in enumerate(names)}

    data = {
        "path": str(OUT.resolve()),  # 루트 경로
        "train": "images/train",
        "val": "images/val",
        "names": names_dict,
    }

    data_yaml_path = OUT / "data.yaml"
    with data_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] data.yaml 생성 완료: {data_yaml_path}")


if __name__ == "__main__":
    prepare_dataset()
