# rename_and_convert_to_jpg.py
import os, re
from PIL import Image, ImageOps, UnidentifiedImageError

# ===== 설정 =====
BASE_DIR = "./data"               # clean, messy 폴더가 있는 상위 경로
CLASSES  = ["clean", "messy"]     # 리네임/변환 대상 폴더
QUALITY  = 92                     # JPG 품질(90~95 권장)
DRY_RUN  = False                  # True면 미리보기(변환/삭제 안 함)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def is_image_file(name: str) -> bool:
    n = name.lower()
    return n.endswith(VALID_EXTS) and not os.path.basename(n).startswith(("._",))

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def to_rgb_jpg(src_path: str, dst_path: str) -> bool:
    """이미지를 RGB JPG로 저장(알파/EXIF 보정 포함). 성공 시 True."""
    with Image.open(src_path) as im:
        # 아이폰/카메라 EXIF 회전 보정
        im = ImageOps.exif_transpose(im)
        # 알파 채널 있으면 흰 배경으로 합성
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        if not DRY_RUN:
            im.save(dst_path, format="JPEG", quality=QUALITY, optimize=True)
    return True

def process_one_folder(folder: str, class_name: str):
    if not os.path.isdir(folder):
        print(f"⚠️ 폴더 없음: {folder}")
        return 0

    files = [f for f in os.listdir(folder) if is_image_file(f)]
    files.sort(key=natural_key)

    count = 0
    idx = 1
    for name in files:
        src = os.path.join(folder, name)
        dst = os.path.join(folder, f"{class_name}_{idx:04d}.jpg")

        try:
            ok = to_rgb_jpg(src, dst)
            if ok and not DRY_RUN:
                # 원본이 이미 .jpg라도 파일명이 다르면 새로 저장 후 원본 삭제
                if os.path.abspath(src) != os.path.abspath(dst):
                    try:
                        os.remove(src)
                    except FileNotFoundError:
                        pass
            count += 1
            idx += 1
        except UnidentifiedImageError:
            print(f"⚠️ 이미지 아님/깨짐: {src}")
        except Exception as e:
            print(f"❗ 처리 실패: {src} -> {dst} ({e})")

    print(f"✅ {class_name}: {count}개 처리 완료 (예: {class_name}_0001.jpg)")
    return count

def main():
    total = 0
    for cls in CLASSES:
        folder = os.path.join(BASE_DIR, cls)
        total += process_one_folder(folder, cls)
    print(f"\n🎯 전체 처리 수: {total}")

if __name__ == "__main__":
    main()
