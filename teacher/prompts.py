"""Teacher prompts and generation settings (FLUX.2 klein 4B image edit, run locally with mflux).

SCENE_PROMPT restyles a whole 512 x 512 photo; FACE_PROMPT restyles an aligned face crop. For faces, the prompt is
extended with a plain-language description of skin, hair, eyes, facial hair, glasses and headwear (from
describe_faces.py): without it the teacher tends to lighten skin tones and change hair.
"""

SEED = 42
STEPS = 4
SIZE = 512

SCENE_PROMPT = (
    "Turn this photo into a screenshot from a stylized 3D life-simulation video game: stylized 3D game characters with "
    "smooth simplified skin and hair, clean plastic-like materials, bright saturated game lighting. Keep exactly the same "
    "framing, composition, people, poses, clothes and objects.")

FACE_PROMPT = (
    "Turn this close-up photo into a screenshot from a stylized 3D life-simulation video game: the person becomes a "
    "stylized 3D game character with smooth simplified skin, simplified 3D hair, clean plastic-like materials, bright "
    "saturated game lighting. This is the same person as in the photo: keep exactly the same framing, head position, "
    "head angle, gaze direction, mouth shape and facial expression, age, face shape, facial hair, hair style and colour, "
    "skin tone, clothes and background layout. Do not add or remove anything.")

DARK_SKINS = ("brown", "dark brown", "very dark")


def parse_description(text: str) -> dict:
    """'key: value; key: value' (describe_faces.py output) -> {key: value}, lower case."""
    d = {}
    for part in text.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            d[k.strip().lower()] = v.strip().strip(".").lower()
    return d


def describe_for_prompt(text: str, ethnicity: str = "") -> str:
    """Plain-language description for the prompt. Only what is present is named (a named attribute always shows up)."""
    d = parse_description(text)
    p = []
    colour, hair = d.get("hair colour", ""), d.get("hair", "")
    if hair == "bald":
        p.append("completely bald smooth head with no hair on top")
    elif hair == "balding":
        p.append(f"balding head with only short thin {colour if 'visible' not in colour else ''} hair on the sides".replace("  ", " "))
    elif colour and "visible" not in colour:
        p.append(f"{colour} hair")
    skin = d.get("skin", "")
    # FairFace ethnicity label (when known) sets a floor on how light the named skin tone can be
    if ethnicity == "Black" and skin not in DARK_SKINS:
        skin = "dark brown"
    if ethnicity == "Indian" and skin not in DARK_SKINS:
        skin = "light brown"
    # light skins: nothing to name (the teacher's default); in-between tones: soft words, otherwise the teacher
    # exaggerates them ("olive" turns khaki)
    skin = {"very fair": "", "fair": "", "light": "", "tan": "lightly tanned warm", "olive": "lightly tanned warm",
            "light brown": "warm light brown", "very dark": "very dark brown"}.get(skin, skin)
    if skin:
        p.append(f"{skin} skin")
    eyes = d.get("eyes", "")
    if eyes and "visible" not in eyes:
        p.append(f"{eyes} eyes")
    moustache, beard = d.get("moustache", "no"), d.get("beard", "no")
    if d.get("person", "man") != "man":
        moustache, beard = "no", "no"
    moustache_colour = moustache[moustache.find("(") + 1:moustache.find(")")] if "(" in moustache else ""
    facial_hair = ""
    if beard.startswith("light stubble"):
        facial_hair = "short stubble beard" + (" and moustache" if moustache.startswith("yes") else "")
    elif not beard.startswith("no"):
        facial_hair = beard.replace("(", "").replace(")", "") + (" with moustache" if moustache.startswith("yes") else "")
    elif moustache.startswith("yes"):
        facial_hair = f"{moustache_colour} moustache and no beard".strip()
    if facial_hair:
        p.append(facial_hair + " (exactly the same facial hair as in the photo, covering the same areas, same length and colour)")
    glasses = d.get("glasses", "no")
    if not glasses.startswith("no"):
        p.append(glasses.replace("(", "").replace(")", ""))
    headwear = d.get("headwear", "no")
    if not headwear.startswith("no") and len(headwear) < 60:
        p.append("wearing " + headwear.replace("what it is and its colour:", "").strip())
    return ", ".join(p)


def face_prompt(description: str) -> str:
    return FACE_PROMPT + (f" The character must have exactly these colours and features, as in the photo: {description}." if description else "")
