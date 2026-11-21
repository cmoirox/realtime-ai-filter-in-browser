"""Select everyday city-life photos from COCO, spread across scene categories.

Reads data/coco/annotations/instances_{train,val}2017.json and writes data/coco/selection.csv
(category, id, url, license, width, height). With --download, also fetches each selected image to
data/sources/coco/<category>/<id>.jpg (resumable; images keep their original Flickr license, see the license column).

    python3 teacher/select_coco.py [--download]
"""
import argparse
import csv
import json
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COCO = ROOT / "data" / "coco"
SOURCES = ROOT / "data" / "sources" / "coco"
random.seed(2026)

FOOD = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
TABLEWARE = {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"}
OFFICE = {"laptop", "keyboard", "mouse"}
KITCHEN = {"oven", "refrigerator", "sink", "microwave", "toaster"}
STREET = {"car", "traffic light", "bicycle", "motorcycle", "stop sign", "parking meter", "fire hydrant", "bench", "truck"}
# Off-topic: sports, zoo or farm animals, planes, boats.
EXCLUDED = {
    "skis", "snowboard", "surfboard", "tennis racket", "baseball bat", "baseball glove", "frisbee", "kite",
    "skateboard", "sports ball", "elephant", "giraffe", "zebra", "bear", "sheep", "cow", "horse", "airplane", "boat",
}

# (name, quota). Order matters: a photo goes to the first category it matches.
QUOTAS = [
    ("metro_train", 300),
    ("bus", 300),
    ("restaurant_meal", 500),
    ("office", 400),
    ("kitchen", 300),
    ("living_room", 350),
    ("bedroom", 150),
    ("street", 600),
    ("group", 250),
    ("portrait", 250),
    ("food_only", 250),
    ("empty_interior", 150),
]


def category(names: set[str], people: list[float]) -> str | None:
    """names = categories present; people = fraction of the image covered by each person."""
    if names & EXCLUDED:
        return None
    n = len(people)
    visible = [p for p in people if p > 0.02]
    if "train" in names and n >= 1:
        return "metro_train"
    if "bus" in names and n >= 1:
        return "bus"
    if "dining table" in names and n >= 1 and names & (FOOD | TABLEWARE):
        return "restaurant_meal"
    if names & OFFICE and n >= 1:
        return "office"
    if names & KITCHEN and n >= 1:
        return "kitchen"
    if names & {"couch", "tv"} and n >= 1:
        return "living_room"
    if "bed" in names and n >= 1:
        return "bedroom"
    if names & STREET and len(visible) >= 1:
        return "street"
    if len(visible) >= 4:
        return "group"
    if 1 <= n <= 3 and max(people) > 0.25:
        return "portrait"
    if n == 0 and names & FOOD and names & (TABLEWARE | {"dining table"}):
        return "food_only"
    if n == 0 and names & (OFFICE | KITCHEN | {"couch", "bed", "dining table", "chair"}):
        return "empty_interior"
    return None


def select() -> None:
    candidates: dict[str, list] = defaultdict(list)
    for split in ("train2017", "val2017"):
        d = json.loads((COCO / "annotations" / f"instances_{split}.json").read_text())
        categories = {c["id"]: c["name"] for c in d["categories"]}
        names: dict[int, set] = defaultdict(set)
        people: dict[int, list] = defaultdict(list)
        images = {i["id"]: i for i in d["images"]}
        for a in d["annotations"]:
            img = images[a["image_id"]]
            name = categories[a["category_id"]]
            names[a["image_id"]].add(name)
            if name == "person" and not a["iscrowd"]:
                people[a["image_id"]].append(a["area"] / (img["width"] * img["height"]))
        for iid, img in images.items():
            # Large enough for a 512 square, and not too elongated (the square crop would cut too much).
            if min(img["width"], img["height"]) < 420 or max(img["width"], img["height"]) / min(img["width"], img["height"]) > 1.6:
                continue
            c = category(names[iid], people[iid])
            if c:
                candidates[c].append((iid, f"http://images.cocodataset.org/{split}/{img['file_name']}", img["license"], img["width"], img["height"]))

    with (COCO / "selection.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "id", "url", "license", "width", "height"])
        total = 0
        for name, quota in QUOTAS:
            batch = candidates[name]
            random.shuffle(batch)
            for row in batch[:quota]:
                w.writerow([name, *row])
            total += min(quota, len(batch))
            print(f"{name:18} {len(batch):6} candidates, {min(quota, len(batch)):4} selected")
        print(f"total selected: {total}")


def fetch(row: dict) -> bool:
    dest = SOURCES / row["category"] / f"{row['id']}.jpg"
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(row["url"], timeout=60) as r:
            dest.write_bytes(r.read())
        return True
    except Exception:
        return False


def download() -> None:
    rows = list(csv.DictReader((COCO / "selection.csv").open()))
    # single images download fine in parallel even though the COCO server is slow per connection
    with ThreadPoolExecutor(16) as pool:
        ok = sum(pool.map(fetch, rows))
    print(f"{ok}/{len(rows)} images in {SOURCES}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="also download the selected images")
    args = ap.parse_args()
    select()
    if args.download:
        download()


if __name__ == "__main__":
    main()
