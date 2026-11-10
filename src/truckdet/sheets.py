"""Images for a person to look at: boxes on images, and contact sheets.

Three checks of the design need eyes: label completeness (100 random images with their boxes), the
scene check (200 pairs of a test image and its nearest train-pool image), and the sample images of
the report. Nothing here is committed with the code: the dataset's licence is CC BY-NC-SA 4.0, so
read it before you publish an image.
"""

from pathlib import Path

from PIL import Image, ImageDraw

from .data.miotcd import CLASSES

GT_COLOUR = (255, 255, 255)
# One colour per arm for detections, in the fixed order of plotting.SERIES, and a fourth for F.
ARM_COLOURS = {"H": (42, 120, 214), "D": (235, 104, 52), "B": (27, 175, 122), "F": (237, 161, 0)}
SHORT = {"articulated_truck": "artic", "bicycle": "bike", "bus": "bus", "car": "car", "motorcycle": "moto",
         "motorized_vehicle": "mv", "non-motorized_vehicle": "nmv", "pedestrian": "ped", "pickup_truck": "pickup",
         "single_unit_truck": "sut", "work_van": "van"}


def _label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, colour) -> None:
    left, top, right, bottom = draw.textbbox((x, y), text)
    draw.rectangle([left - 1, top - 1, right + 1, bottom + 1], fill=(0, 0, 0))
    draw.text((x, y), text, fill=colour)


def draw_boxes(image: Image.Image, boxes, labels, colour=GT_COLOUR, width: int = 2, below: bool = False) -> Image.Image:
    """Draw xyxy `boxes` with text `labels` on a copy of `image`. `below`: the text goes under the box."""
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for (x0, y0, x1, y1), text in zip(boxes, labels):
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=width)
        if text:
            _label(draw, x0 + 2, y1 + 2 if below else max(0, y0 - 12), text, colour)
    return image


def draw_ground_truth(image: Image.Image, annotations: list[dict]) -> Image.Image:
    boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in annotations]
    return draw_boxes(image, boxes, [SHORT[CLASSES[a["category_id"] - 1]] for a in annotations])


def contact_sheet(tiles: list[Image.Image], captions: list[str], columns: int, tile_size=(320, 240),
                  out_path=None) -> Image.Image:
    """Tiles (resized to fit `tile_size`) in a grid with a caption under each."""
    width, height = tile_size
    caption_height = 16
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + caption_height)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for k, (tile, caption) in enumerate(zip(tiles, captions)):
        tile = tile.copy()
        tile.thumbnail(tile_size)
        x, y = (k % columns) * width, (k // columns) * (height + caption_height)
        sheet.paste(tile, (x + (width - tile.width) // 2, y + (height - tile.height) // 2))
        draw.text((x + 4, y + height + 2), caption, fill=(230, 230, 230))
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_path, quality=90)
    return sheet


def label_sheets(coco: dict, image_ids, image_dir, out_dir, per_sheet: int = 20) -> list[Path]:
    """The label completeness check: each image of `image_ids` with its boxes, `per_sheet` to a sheet."""
    images = {im["id"]: im for im in coco["images"]}
    anns: dict[int, list[dict]] = {int(i): [] for i in image_ids}
    for a in coco["annotations"]:
        if a["image_id"] in anns:
            anns[a["image_id"]].append(a)
    out = []
    ids = [int(i) for i in image_ids]
    for start in range(0, len(ids), per_sheet):
        tiles, captions = [], []
        for i in ids[start : start + per_sheet]:
            with Image.open(Path(image_dir) / images[i]["file_name"]) as im:
                tiles.append(draw_ground_truth(im, anns[i]))
            captions.append(f"{images[i]['miotcd_id']}  {len(anns[i])} boxes")
        path = Path(out_dir) / f"labels_{start // per_sheet + 1:02d}.jpg"
        contact_sheet(tiles, captions, columns=4, tile_size=(480, 360), out_path=path)
        out.append(path)
    return out


def pair_sheets(pairs: list[tuple[str, str, float]], image_dir, out_dir, prefix: str, per_sheet: int = 25) -> list[Path]:
    """The scene check: each pair (test image ID, train-pool image ID, similarity) side by side."""
    out = []
    for start in range(0, len(pairs), per_sheet):
        tiles, captions = [], []
        for k, (a, b, similarity) in enumerate(pairs[start : start + per_sheet], start=start + 1):
            for image_id, role in ((a, "test"), (b, "train pool")):
                with Image.open(Path(image_dir) / f"{image_id}.jpg") as im:
                    tiles.append(im.convert("RGB"))
                captions.append(f"pair {k} {role} {image_id}  sim {similarity:.3f}")
        path = Path(out_dir) / f"{prefix}_{start // per_sheet + 1:02d}.jpg"
        contact_sheet(tiles, captions, columns=2, tile_size=(360, 270), out_path=path)
        out.append(path)
    return out


def detection_image(image: Image.Image, annotations: list[dict], by_arm: dict[str, tuple], min_score: float = 0.5) -> Image.Image:
    """Ground truth (white, label above) and each arm's detections (its colour, label below). `by_arm`: arm ->
    (scores [K, C], boxes [K, 4] xyxy) of the image's queries; a query is drawn with its top class if that
    score is at least `min_score`."""
    out = draw_ground_truth(image, annotations)
    for arm, (scores, boxes) in by_arm.items():
        keep = scores.max(-1) >= min_score
        labels = [f"{arm} {SHORT[CLASSES[c]]} {s:.2f}" for c, s in zip(scores[keep].argmax(-1), scores[keep].max(-1))]
        out = draw_boxes(out, boxes[keep].tolist(), labels, colour=ARM_COLOURS.get(arm, (255, 0, 255)), width=1, below=True)
    return out
