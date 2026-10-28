"""출력 담당: 동물 박스·품종 표시, 이미지와 CSV 저장.
설명할 함수: annotate, save_image, write_csv.
실습: DOG_COLOR와 CAT_COLOR를 바꾸면 박스 색상이 바뀝니다.
"""

DOG_COLOR = (50, 210, 120)
CAT_COLOR = (80, 180, 255)
def animal_counts(predictions):
    return {"dog_count": sum(p["species"] == "dog" for p in predictions),
            "cat_count": sum(p["species"] == "cat" for p in predictions)}


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
    if any(p.get("reason") == "single_pet_assumed_not_yolo_detected" for p in predictions):
        title += "  |  REVIEW: NOT YOLO DETECTED"
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
