"""Stitch reference images into labeled master contact sheets."""
from PIL import Image, ImageDraw, ImageFont
import os

REF_DIR = "/home/user/CartoonStyle/examples/apoc_day1_video/style_tests"
OUT_DIR = REF_DIR

# ---- Character sheet: 3 cols x 4 rows = 12 panels ----
char_panels = [
    ("MC - full body front",       "ref_mc_fullbody.png"),
    ("MC - behind counter",        "ref_mc_counter.png"),
    ("MC - with baseball bat",     "ref_mc_bat.png"),
    ("Zombie - base body front",   "ref_zombie_base.png"),
    ("Zombie - sprint side view",  "ref_zombie_sprint.png"),
    ("Zombie - empty sleeve",      "ref_zombie_missingarm.png"),
    ("Zombie - jaw injury",        "ref_zombie_jaw.png"),
    ("Zombies at glass (exterior)","ref_zombies_at_glass.png"),
    ("BLANK",""),("BLANK",""),("BLANK",""),("BLANK",""),
]

# ---- Location sheet: 3 cols x 2 rows = 6 panels ----
loc_panels = [
    ("Exterior pristine",  "ref_store_pristine.png"),
    ("Interior empty",     "ref_interior_empty.png"),
    ("Back aisle/cooler",  "ref_back_aisle.png"),
    ("Cooler interior",    "ref_cooler_interior.png"),
    ("Storage room",       "ref_storage_room.png"),
    ("Style A texture",    "styleA_highcontrast.png"),
]

PANEL_W, PANEL_H = 768, 768     # each panel (square)
LABEL_H = 60                    # label strip at bottom of each panel
COLS = 3
PAD = 20                        # padding between panels and outer margin
LABEL_FONT_SIZE = 28
TITLE_FONT_SIZE = 48
TITLE_H = 90


def build_sheet(panels, title, cols=COLS, save_as="sheet.png"):
    rows = (len(panels) + cols - 1) // cols
    cell_w = PANEL_W + PAD
    cell_h = PANEL_H + LABEL_H + PAD
    sheet_w = PAD + cols * cell_w
    sheet_h = PAD + TITLE_H + rows * cell_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), (240, 240, 240))
    draw = ImageDraw.Draw(sheet)

    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", LABEL_FONT_SIZE)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", TITLE_FONT_SIZE)
    except Exception:
        font_label = ImageFont.load_default()
        font_title = ImageFont.load_default()

    # Title
    draw.text((PAD, PAD), title, fill=(0,0,0), font=font_title)

    for i, (label, fname) in enumerate(panels):
        r, c = divmod(i, cols)
        x = PAD + c * cell_w
        y = PAD + TITLE_H + r * cell_h
        if fname:
            img_path = os.path.join(REF_DIR, fname)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                # Fit image into PANEL_W x PANEL_H preserving aspect
                img.thumbnail((PANEL_W, PANEL_H), Image.LANCZOS)
                # Center
                ox = x + (PANEL_W - img.width) // 2
                oy = y + (PANEL_H - img.height) // 2
                # Draw panel bg white behind image
                draw.rectangle([x, y, x + PANEL_W, y + PANEL_H + LABEL_H], fill=(255,255,255), outline=(0,0,0), width=2)
                sheet.paste(img, (ox, oy))
            else:
                draw.rectangle([x, y, x+PANEL_W, y+PANEL_H+LABEL_H], fill=(220,220,220), outline=(0,0,0), width=2)
                draw.text((x+10, y+PANEL_H//2-10), f"MISSING: {fname}", fill=(180,0,0), font=font_label)
        else:
            draw.rectangle([x, y, x+PANEL_W, y+PANEL_H+LABEL_H], fill=(230,230,230), outline=(180,180,180), width=1)
        # Label strip
        label_y = y + PANEL_H + 8
        draw.text((x + 10, label_y), label, fill=(0,0,0), font=font_label)

    out = os.path.join(REF_DIR, save_as)
    sheet.save(out, "PNG")
    print(f"Wrote {out} ({sheet_w}x{sheet_h})")


build_sheet(char_panels, "CHARACTER MODEL SHEET — CartoonStyle Video 5", save_as="sheet_characters.png")
build_sheet(loc_panels,    "LOCATION MODEL SHEET — CartoonStyle Video 5",  save_as="sheet_locations.png")
