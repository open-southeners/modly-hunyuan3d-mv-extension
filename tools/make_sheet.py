"""
Build a 2x2 view sheet for the Hunyuan3D 2 MV Modly extension.

    python make_sheet.py --front front.jpg [--left left.jpg] [--back back.jpg] [--right right.jpg] -o sheet.png

Layout:  front | left
         back  | right

Each photo is fitted (aspect kept) into a square cell on a white background;
views you leave out stay blank and the extension skips them. Needs Pillow only.
"""
import argparse

from PIL import Image

CELLS = {"front": (0, 0), "left": (1, 0), "back": (0, 1), "right": (1, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--front", required=True)
    parser.add_argument("--left")
    parser.add_argument("--back")
    parser.add_argument("--right")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--cell", type=int, default=1024, help="cell size in pixels (default 1024)")
    args = parser.parse_args()

    sheet = Image.new("RGB", (args.cell * 2, args.cell * 2), "white")
    for view, (col, row) in CELLS.items():
        path = getattr(args, view)
        if not path:
            continue
        img = Image.open(path)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            flat = Image.new("RGB", img.size, "white")
            flat.paste(img, mask=img.getchannel("A"))
            img = flat
        else:
            img = img.convert("RGB")
        img.thumbnail((args.cell, args.cell), Image.LANCZOS)
        x = col * args.cell + (args.cell - img.width) // 2
        y = row * args.cell + (args.cell - img.height) // 2
        sheet.paste(img, (x, y))

    sheet.save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
