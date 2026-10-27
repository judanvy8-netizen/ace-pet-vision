"""출력 담당: 동물 박스·품종 표시, 이미지와 CSV 저장.
설명할 함수: annotate, save_image, write_csv.
실습: DOG_COLOR와 CAT_COLOR를 바꾸면 박스 색상이 바뀝니다.
"""
import csv

DOG_COLOR = (50, 210, 120)
CAT_COLOR = (80, 180, 255)
FIELDS = ["filename", "object_id", "species", "breed", "candidate_breed", "breed_confidence",
          "detection_confidence", "status", "reason", "x1", "y1", "x2", "y2", "top3_json",
          "result_image", "image_processing_ms"]
SUMMARY_FIELDS = ["filename", "cat", "dog", "actual_cat", "actual_dog", "cat_error", "dog_error",
                  "status", "animal_count", "processing_ms", "error", "result_image"]


def animal_counts(predictions):
    return {"dog_count": sum(p["species"] == "dog" for p in predictions),
            "cat_count": sum(p["species"] == "cat" for p in predictions)}


def write_csv(path, rows, fields):
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def annotate(image, predictions):
    from PIL import Image, ImageDraw, ImageFont
    result = image.copy()
    draw = ImageDraw.Draw(result)
    size = max(12, min(28, round(image.width / 50)))
    font = ImageFont.load_default(size=size)
    for number, prediction in enumerate(predictions, 1):
        x1, y1, x2, y2 = (prediction[k] for k in ("x1", "y1", "x2", "y2"))
        color = DOG_COLOR if prediction["species"] == "dog" else CAT_COLOR
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=max(2, round(image.width / 500)))
        label = f'#{number} {prediction["species"]}: {prediction["breed"]} {prediction["breed_confidence"]:.0%}'
        if prediction["status"] == "review":
            label = f'#{number} {prediction["species"]}: {prediction["candidate_breed"]} {prediction["breed_confidence"]:.0%} (review)'
        current_font = font
        while draw.textbbox((0, 0), label, font=current_font)[2] > image.width - 8 and size > 8:
            size -= 1
            current_font = ImageFont.load_default(size=size)
        box = draw.textbbox((0, 0), label, font=current_font)
        width, height = box[2] + 8, box[3] - box[1] + 8
        x = min(x1, max(0, image.width - width))
        y = max(0, y1 - height)
        draw.rectangle((x, y, min(image.width - 1, x + width), y + height), fill=color)
        draw.text((x + 4, y + 4 - box[1]), label, font=current_font, fill=(0, 0, 0))
    if not predictions:
        draw.text((8, 8), "No cat/dog detected", font=font, fill=(255, 70, 70), stroke_width=1, stroke_fill=(0, 0, 0))
    # 원본 위에 여백을 추가하여 동물을 가리지 않고 사진별 마릿수를 표시합니다.
    counts = animal_counts(predictions)
    title = f'Dogs: {counts["dog_count"]}  |  Cats: {counts["cat_count"]}'
    title_size = max(8, min(28, image.width // 12))
    title_font = ImageFont.load_default(size=title_size)
    while draw.textbbox((0, 0), title, font=title_font)[2] > image.width - 8 and title_size > 1:
        title_size -= 1
        title_font = ImageFont.load_default(size=title_size)
    header_height = title_size + 20
    canvas = Image.new("RGB", (image.width, image.height + header_height), (25, 30, 40))
    canvas.paste(result, (0, header_height))
    ImageDraw.Draw(canvas).text((4, 6), title, font=title_font, fill="white")
    return canvas



def save_image(image, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(destination.suffix, bgr)
    if not success:
        raise OSError(f"이미지 인코딩 실패: {destination}")
    encoded.tofile(destination)
