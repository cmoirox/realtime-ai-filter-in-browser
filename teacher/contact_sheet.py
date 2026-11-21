"""Contact sheet for human review: tiles photo | styled pairs (photo left, styled right) into one JPEG.

Names come from a split list (data/splits/<list>.txt, or any text file with one name per line), or from a
verdict.csv filtered by --reason. Faces are looked up in data/faces, scenes in data/scenes (by list name prefix,
or --set).

    uv run --python 3.12 --with pillow python teacher/contact_sheet.py scenes_test [--n 20] [--shuffle] [--out sheet.jpg]
    uv run --python 3.12 --with pillow python teacher/contact_sheet.py data/scenes/verdict.csv --reason invented_person
"""
import argparse
import csv
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
T, COLS = 256, 4


def sheet(folder: Path, names: list[str], out: Path) -> None:
    rows = (len(names) + COLS - 1) // COLS
    im = Image.new("RGB", (COLS * (2 * T + 12), rows * (T + 12)), "white")
    for i, n in enumerate(names):
        x, y = (i % COLS) * (2 * T + 12), (i // COLS) * (T + 12)
        im.paste(Image.open(folder / "photo" / f"{n}.png").convert("RGB").resize((T, T), Image.LANCZOS), (x, y))
        im.paste(Image.open(folder / "styled" / f"{n}.png").convert("RGB").resize((T, T), Image.LANCZOS), (x + T, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=88)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="split name (e.g. faces_val), a .txt list, or a verdict.csv")
    ap.add_argument("--reason", help="with a verdict.csv: only rows with this rejection reason ('' = kept)")
    ap.add_argument("--set", choices=("scenes", "faces"), help="image folder (default: guessed from the source name)")
    ap.add_argument("--n", type=int, default=20, help="number of pairs")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--out", help="output JPEG (default: data/contact_sheets/<set>_<source>.jpg)")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        src = ROOT / "data" / "splits" / f"{args.source}.txt"
    if src.suffix == ".csv":
        names = [r["name"] for r in csv.DictReader(src.open()) if args.reason is None or r["reason"] == args.reason]
    else:
        names = src.read_text().split()
    if args.shuffle:
        random.Random(14).shuffle(names)
    subset = args.set or ("faces" if src.stem.startswith("faces") or src.parent.name == "faces" else "scenes")
    tag = src.stem + (f"_{args.reason}" if args.reason else "")
    if not tag.startswith(subset):
        tag = f"{subset}_{tag}"
    out = Path(args.out) if args.out else ROOT / "data" / "contact_sheets" / f"{tag}.jpg"
    sheet(ROOT / "data" / subset, names[:args.n], out)


if __name__ == "__main__":
    main()
