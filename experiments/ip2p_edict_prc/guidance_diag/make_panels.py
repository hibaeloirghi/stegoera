"""Build the fixed panel order from images the case jobs already saved.

Order: source, reference edit, G1, G2, G3, G4. A missing config stays a blank tile.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "ip2p_edict_prc" / "guidance_diag"
ORDER = ("source", "reference", "g1", "g2", "g3", "g4")


def main() -> int:
    table = OUT / "edit_quality.csv"
    if not table.exists():
        print("no edit_quality.csv yet")
        return 2
    with table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_edit: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_edit.setdefault(row["edit"], {})[row["role"]] = row
    for edit, roles in by_edit.items():
        if edit.startswith("dog1"):
            continue
        tiles = []
        labels = []
        for role in ORDER:
            row = roles.get(role)
            path = ROOT / row["image_path"] if row and row.get("image_path") else None
            if path is not None and path.exists():
                image = Image.open(path).convert("RGB")
            else:
                image = Image.new("RGB", (512, 512), (220, 220, 220))
            if image.size != (512, 512):
                image = image.resize((512, 512))
            tiles.append(image)
            labels.append(role if row is None else f"{role} t={row.get('guidance_text','')} i={row.get('guidance_image','')}")
        label_h = 36
        sheet = Image.new("RGB", (512 * len(tiles), 512 + label_h), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (image, label) in enumerate(zip(tiles, labels)):
            sheet.paste(image, (index * 512, label_h))
            draw.text((index * 512 + 8, 8), label, fill="black")
        destination = OUT / "plots" / f"panel_{edit}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination)
        print("wrote", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
