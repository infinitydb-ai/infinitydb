from PIL import Image, ImageDraw, ImageFont

# 1000 x 420 Cover Image for Dev.to
w, h = 1000, 420
img = Image.new("RGBA", (w, h), (10, 13, 20, 255))
draw = ImageDraw.Draw(img)

# Background ambient glow
for r in range(250, 0, -3):
    alpha = int(30 * (1 - r / 250))
    draw.ellipse((150 - r, 120 - r, 150 + r, 120 + r), fill=(59, 130, 246, alpha))
    draw.ellipse((850 - r, 300 - r, 850 + r, 300 + r), fill=(16, 185, 129, alpha))

try:
    font_title = ImageFont.truetype("arialbd.ttf", 46)
    font_sub = ImageFont.truetype("arial.ttf", 20)
    font_tag = ImageFont.truetype("arialbd.ttf", 15)
    font_pill = ImageFont.truetype("arialbd.ttf", 14)
except:
    font_title = ImageFont.load_default()
    font_sub = ImageFont.load_default()
    font_tag = ImageFont.load_default()
    font_pill = ImageFont.load_default()

# Pill badge
draw.rounded_rectangle((70, 48, 230, 80), radius=16, fill=(59, 130, 246, 40), outline=(59, 130, 246, 120))
draw.text((86, 56), "OPEN SOURCE AI", fill=(96, 165, 250, 255), font=font_tag)

# Title
draw.text((70, 105), "Building a Sub-Millisecond", fill=(255, 255, 255, 255), font=font_title)
draw.text((70, 160), "Agent Memory Engine", fill=(59, 130, 246, 255), font=font_title)

# Subtitle
draw.text((70, 235), "Tiered L0-L6 Neurosymbolic Architecture on SQLite WAL Mode", fill=(148, 163, 184, 255), font=font_sub)

# Metric pill badges at bottom
metrics = [
    ("RECALL", "0.75ms"),
    ("TOKEN SAVINGS", "81.4%"),
    ("EXTRACTION COST", "$0"),
    ("AUDIT PROOFS", "SHA-256")
]

mx = 70
for lbl, val in metrics:
    bw = 180
    draw.rounded_rectangle((mx, 320, mx + bw, 370), radius=10, fill=(18, 24, 38, 220), outline=(255, 255, 255, 30))
    draw.text((mx + 14, 335), f"{lbl}:", fill=(148, 163, 184, 255), font=font_pill)
    draw.text((mx + 115 if len(lbl) > 7 else mx + 85, 335), val, fill=(52, 211, 153, 255), font=font_pill)
    mx += bw + 20

img.convert("RGB").save("assets/devto_cover.png", "PNG")
print("[+] Created assets/devto_cover.png (1000x420)")
