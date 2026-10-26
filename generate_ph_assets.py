import os
import math
from PIL import Image, ImageDraw, ImageFont

os.makedirs("assets", exist_ok=True)

# 1. Generate 512x512 Square Thumbnail for Product Hunt (240x240 recommended)
thumb_size = 512
img = Image.new("RGBA", (thumb_size, thumb_size), (10, 10, 10, 255))
draw = ImageDraw.Draw(img)

# Background subtle radial glow
center = (thumb_size // 2, thumb_size // 2 - 30)
for r in range(180, 0, -2):
    alpha = int(25 * (1 - r / 180))
    draw.ellipse(
        (center[0] - r, center[1] - r, center[0] + r, center[1] + r),
        fill=(59, 130, 246, alpha)
    )

# Draw Infinity Symbol in Center
points = []
scale = 120
for t in range(0, 360, 2):
    rad = math.radians(t)
    denom = 1 + math.sin(rad) ** 2
    x = center[0] + (scale * math.cos(rad)) / denom
    y = center[1] + (scale * math.sin(rad) * math.cos(rad)) / denom
    points.append((x, y))

# Draw smooth path
for i in range(len(points)):
    p1 = points[i]
    p2 = points[(i + 1) % len(points)]
    draw.line([p1, p2], fill=(255, 255, 255, 255), width=10)

# Accent nodes on infinity loop
node_scale = scale * 0.7
left_node = (center[0] - 65, center[1])
right_node = (center[0] + 65, center[1])
draw.ellipse((left_node[0]-14, left_node[1]-14, left_node[0]+14, left_node[1]+14), fill=(59, 130, 246, 255))
draw.ellipse((right_node[0]-14, right_node[1]-14, right_node[0]+14, right_node[1]+14), fill=(59, 130, 246, 255))

# Draw text
try:
    font_bold = ImageFont.truetype("arialbd.ttf", 46)
    font_sub = ImageFont.truetype("arial.ttf", 18)
except:
    font_bold = ImageFont.load_default()
    font_sub = ImageFont.load_default()

title = "InfinityDB"
bbox = draw.textbbox((0, 0), title, font=font_bold)
w = bbox[2] - bbox[0]
draw.text(((thumb_size - w) // 2, 365), title, fill=(255, 255, 255, 255), font=font_bold)

subtitle = "AGENT MEMORY ENGINE"
bbox_sub = draw.textbbox((0, 0), subtitle, font=font_sub)
w_sub = bbox_sub[2] - bbox_sub[0]
draw.text(((thumb_size - w_sub) // 2, 425), subtitle, fill=(156, 163, 175, 255), font=font_sub)

img.convert("RGB").save("assets/thumbnail.png", "PNG")
print("[+] Created assets/thumbnail.png (512x512)")


# 2. Generate 1280x760 Gallery Preview Image
gal_w, gal_h = 1280, 760
gal = Image.new("RGBA", (gal_w, gal_h), (9, 13, 22, 255))
draw_gal = ImageDraw.Draw(gal)

# Subtle background gradients
for r in range(400, 0, -4):
    alpha = int(35 * (1 - r / 400))
    draw_gal.ellipse((200 - r, 200 - r, 200 + r, 200 + r), fill=(59, 130, 246, alpha))
    draw_gal.ellipse((1080 - r, 560 - r, 1080 + r, 560 + r), fill=(16, 185, 129, alpha))

try:
    font_gal_title = ImageFont.truetype("arialbd.ttf", 52)
    font_gal_sub = ImageFont.truetype("arial.ttf", 24)
    font_card_val = ImageFont.truetype("arialbd.ttf", 42)
    font_card_lbl = ImageFont.truetype("arial.ttf", 16)
    font_card_sub = ImageFont.truetype("arial.ttf", 14)
except:
    font_gal_title = ImageFont.load_default()
    font_gal_sub = ImageFont.load_default()
    font_card_val = ImageFont.load_default()
    font_card_lbl = ImageFont.load_default()
    font_card_sub = ImageFont.load_default()

# Header Brand
draw_gal.text((80, 70), "InfinityDB", fill=(255, 255, 255, 255), font=font_gal_title)
draw_gal.text((80, 135), "Sub-Millisecond Neurosymbolic Long-Term Memory for AI Agents", fill=(148, 163, 184, 255), font=font_gal_sub)

# 4 Metric Cards
cards = [
    ("RECALL LATENCY (p95)", "0.75 ms", "< 5.0ms target", (59, 130, 246)),
    ("TOKEN COMPRESSION", "81.4 %", "Context overhead saved", (16, 185, 129)),
    ("FACT ACCURACY", "89.7 %", "PersonaMem Benchmark", (139, 92, 246)),
    ("EXTRACTION COST", "$0", "Deterministic local rules", (245, 158, 11)),
]

card_w, card_h = 260, 160
start_x, start_y = 80, 210
gap = 26

for idx, (lbl, val, sub, border_col) in enumerate(cards):
    cx = start_x + idx * (card_w + gap)
    cy = start_y
    draw_gal.rounded_rectangle(
        (cx, cy, cx + card_w, cy + card_h),
        radius=14,
        fill=(18, 24, 38, 220),
        outline=(255, 255, 255, 25),
        width=1
    )
    draw_gal.text((cx + 20, cy + 22), lbl, fill=(148, 163, 184, 255), font=font_card_lbl)
    draw_gal.text((cx + 20, cy + 55), val, fill=(255, 255, 255, 255), font=font_card_val)
    draw_gal.text((cx + 20, cy + 115), sub, fill=(52, 211, 153, 255), font=font_card_sub)

# Features Banner Box
feat_y = 410
feat_w = 1120
feat_h = 270
draw_gal.rounded_rectangle(
    (start_x, feat_y, start_x + feat_w, feat_y + feat_h),
    radius=16,
    fill=(15, 23, 42, 200),
    outline=(255, 255, 255, 30),
    width=1
)

draw_gal.text((start_x + 30, feat_y + 25), "L0-L6 Neurosymbolic Architecture", fill=(255, 255, 255, 255), font=font_gal_sub)

layers = [
    ("L0 Raw Logs", "Disk-backed append-only interaction store"),
    ("L1 Atomic Facts", "Zero-LLM deterministic extraction (<10ms)"),
    ("L2 Episodic Graph", "Causal relationship tracking between events"),
    ("L3 Dynamic Persona", "Chronological provenance conflict resolution"),
    ("L4 Policies", "Corporate SOPs & safety guardrail injection"),
    ("L6 Proof Chain", "Tamper-evident SHA-256 audit provenance"),
]

for l_idx, (layer_title, layer_desc) in enumerate(layers):
    col = l_idx % 2
    row = l_idx // 2
    lx = start_x + 35 + col * 550
    ly = feat_y + 80 + row * 55
    draw_gal.text((lx, ly), f"• {layer_title}:", fill=(96, 165, 250, 255), font=font_card_lbl)
    draw_gal.text((lx + 175, ly), layer_desc, fill=(203, 213, 225, 255), font=font_card_lbl)

gal.convert("RGB").save("assets/gallery_overview.png", "PNG")
print("[+] Created assets/gallery_overview.png (1280x760)")
