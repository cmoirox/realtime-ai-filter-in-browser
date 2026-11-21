"""Complete the scene sources with freely licensed photos from Wikimedia Commons, for the categories COCO
covers poorly (inside public transport, the Paris metro, cafes, shops...).

Downloads a 1024 px thumbnail into data/sources/wikimedia/<category>/<pageid>.jpg and records the author and
license of every image in data/sources/wikimedia/credits.csv (category, file, title, license, author, page).
Resumable: files already listed in credits.csv are skipped.

    python3 teacher/download_wikimedia.py
"""
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "sources" / "wikimedia"
API = "https://commons.wikimedia.org/w/api.php"
AGENT = {"User-Agent": "dataset-builder/0.1 (non-commercial research project)"}

# category -> (search queries, target count).
# Some queries are deliberately in French or German: Commons file titles and descriptions are multilingual,
# and local-language queries find photos the English ones miss.
SEARCHES = {
    "metro_paris": ([
        "métro de Paris intérieur rame voyageurs", "Paris Métro interior train passengers", "MF 01 intérieur",
        "MP 05 intérieur", "MF 77 intérieur", "MP 14 intérieur", "métro Paris quai voyageurs station",
        "Paris metro platform passengers", "RER intérieur voyageurs", "métro Paris couloir correspondance",
        "Paris métro station entrance people", "tramway Île-de-France intérieur",
    ], 200),
    "metro_interior": (["metro train interior passengers", "subway car interior", "inside metro carriage people", "interior of underground train", "U-Bahn Innenraum"], 120),
    "bus_interior": (["bus interior passengers", "inside city bus", "interior of a bus seats", "intérieur bus RATP", "Bus Innenraum Fahrgäste"], 120),
    "tram_interior": (["tram interior passengers", "inside tramway", "Straßenbahn Innenraum"], 50),
    "station_platform": (["metro station platform passengers", "subway platform people waiting"], 50),
    "cafe_restaurant": (["cafe interior customers", "restaurant interior people dining", "coffee shop interior people", "bar interior people", "terrasse café Paris"], 90),
    "open_plan_office": (["open plan office people working", "office interior desks employees", "coworking space people"], 60),
    "shop": (["supermarket interior shoppers", "shop interior customers", "bakery interior counter", "boulangerie intérieur"], 60),
    "street_pedestrians": (["pedestrians city street", "people walking sidewalk city", "crosswalk pedestrians", "rue Paris piétons"], 80),
}


def call(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({"format": "json", **params})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {}


def search(query: str, limit: int) -> list[dict]:
    d = call({
        "action": "query", "generator": "search", "gsrnamespace": 6, "gsrlimit": limit,
        "gsrsearch": f"{query} filetype:bitmap filew:>900 fileh:>700",
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1024,
        "iiextmetadatafilter": "LicenseShortName|Artist",
    })
    return list(d.get("query", {}).get("pages", {}).values())


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    credits = DEST / "credits.csv"
    seen = set()
    if credits.exists():
        seen = {r["file"] for r in csv.DictReader(credits.open())}
    with credits.open("a", newline="") as f:
        w = csv.writer(f)
        if not seen:
            w.writerow(["category", "file", "title", "license", "author", "page"])
        for category, (queries, target) in SEARCHES.items():
            folder = DEST / category
            folder.mkdir(exist_ok=True)
            taken = len(list(folder.glob("*.jpg")))
            for query in queries:
                if taken >= target:
                    break
                for page in search(query, 50):
                    if taken >= target:
                        break
                    info = (page.get("imageinfo") or [{}])[0]
                    name = f"{page.get('pageid')}.jpg"
                    if info.get("mime") != "image/jpeg" or name in seen or not info.get("thumburl"):
                        continue
                    w_, h_ = info.get("width", 0), info.get("height", 0)
                    if not w_ or max(w_, h_) / min(w_, h_) > 1.8:  # panoramas: the square crop would cut too much
                        continue
                    try:
                        req = urllib.request.Request(info["thumburl"], headers=AGENT)
                        with urllib.request.urlopen(req, timeout=60) as r:
                            (folder / name).write_bytes(r.read())
                    except Exception:
                        continue
                    meta = info.get("extmetadata", {})
                    w.writerow([category, name, page.get("title"), meta.get("LicenseShortName", {}).get("value", ""),
                                meta.get("Artist", {}).get("value", "")[:200], info.get("descriptionurl", "")])
                    f.flush()
                    seen.add(name)
                    taken += 1
                    time.sleep(0.15)
            print(f"{category:20} {taken}", flush=True)


if __name__ == "__main__":
    main()
