"""Faces: describe every face (skin, hair, eyes, beard, glasses, headwear) with a small local vision-language model
(Qwen2.5-VL 7B, 4-bit, MLX), so that the teacher prompt NAMES these attributes (otherwise the teacher lightens skin
tones and changes hair).

    uv run --python 3.12 --with mlx-vlm --with pillow python teacher/describe_faces.py [photo folder] [output.csv] [--view]
    uv run --python 3.12 --with mlx-vlm --with pillow python teacher/describe_faces.py --profiles
Default: data/faces/photo -> data/faces/descriptions.csv (name, description). Resumable.
--view also asks which way the head is turned. --profiles describes the profile crops listed in
data/faces/profiles/selected.csv into data/faces/profiles/descriptions.csv (implies --view).
"""
import csv
import sys
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

ROOT = Path(__file__).resolve().parent.parent
FACES = ROOT / "data" / "faces"
MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
QUESTION = (
    "Look closely at the main person's face in this photo. Judge the real skin colour, not the lighting. "
    "Answer with exactly one line in this format, choosing one option for each item, nothing else:\n"
    "person: <man | woman | boy | girl | baby>; "
    "hair colour: <black | dark brown | brown | light brown | blond | red | grey | white | dyed (say colour) | not visible>; "
    "hair: <bald | balding | shaved very short | short | medium | long | tied back | not visible>; "
    "skin: <very fair | fair | light | tan | olive | light brown | brown | dark brown | very dark>; "
    "eyes: <blue | green | grey | hazel | brown | dark brown | not visible>; "
    "moustache: <no | yes (colour)>; "
    "beard: <no | light stubble | short beard | full beard | goatee> (colour if any); "
    "glasses: <no | glasses with (colour) frame | sunglasses>; "
    "headwear: <no | what it is and its colour>")
VIEW_QUESTION = "; view of the head: <front | three-quarter | side profile | from behind>"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    profiles = "--profiles" in sys.argv
    if profiles:
        folder, out = FACES / "profiles" / "photo", FACES / "profiles" / "descriptions.csv"
        wanted = {r["name"] for r in csv.DictReader((FACES / "profiles" / "selected.csv").open())}
    else:
        folder = Path(args[0]) if len(args) > 0 else FACES / "photo"
        out = Path(args[1]) if len(args) > 1 else FACES / "descriptions.csv"
        wanted = None
    question = QUESTION + (VIEW_QUESTION if profiles or "--view" in sys.argv else "")
    done = {r[0] for r in csv.reader(out.open())} if out.exists() else set()
    model, proc = load(MODEL)
    prompt = apply_chat_template(proc, load_config(MODEL), question, num_images=1)
    photos = sorted(p for p in folder.glob("*.png") if p.stem not in done and (wanted is None or p.stem in wanted))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        for k, p in enumerate(photos):
            r = generate(model, proc, prompt, [str(p)], max_tokens=120, verbose=False, temperature=0.0)
            text = (r.text if hasattr(r, "text") else str(r)).strip().split("\n")[0]
            w.writerow([p.stem, text])
            f.flush()
            if k % 100 == 0:
                print(k, len(photos), p.stem, text, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
