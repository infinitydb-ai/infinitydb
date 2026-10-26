import math
from PIL import Image, ImageDraw, ImageFont

def draw_infinity_emblem(draw, ox, oy, scale=1.0):
    """Draw the exact emblem from assets/logo.svg"""
    # Cubic bezier points for the infinity loop
    # M12 36 C 12 14, 32 14, 38 36 C 44 58, 64 58, 64 36 C 64 14, 44 14, 38 36 Z
    def bezier_point(p0, p1, p2, p3, t):
        x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
        return (ox + x * scale, oy + y * scale)

    # First loop
    pts1 = [bezier_point((12, 36), (12, 14), (32, 14), (38, 36), t/50.0) for t in range(51)]
    # Second loop
    pts2 = [bezier_point((38, 36), (44, 58), (64, 58), (64, 36), t/50.0) for t in range(51)]
    # Third loop
    pts3 = [bezier_point((64, 36), (64, 14), (44, 14), (38, 36), t/50.0) for t in range(51)]
    # Fourth loop
    pts4 = [bezier_point((38, 36), (32, 58), (12, 58), (12, 36), t/50.0) for t in range(51)]

    all_pts = pts1 + pts2 + pts3 + pts4
    for i in range(len(all_pts) - 1):
        draw.line([all_pts[i], all_pts[i+1]], fill=(255, 255, 255, 255), width=int(3.4 * scale))

    # Accent blue and white circles
    def draw_c(cx, cy, r, color):
        rx, ry = ox + cx * scale, oy + cy * scale
        rr = r * scale
        draw.ellipse((rx - rr, ry - rr, rx + rr, ry + rr), fill=color)

    draw_c(22, 36, 4.2, (59, 130, 246, 255))
    draw_c(54, 36, 4.2, (59, 130, 246, 255))
    draw_c(38, 20, 3.0, (255, 255, 255, 220))
    draw_c(38, 52, 3.0, (255, 255, 255, 220))


# 1. Exact README Logo (Landscape PNG: 1040 x 280)
w, h = 1040, 280
img_logo = Image.new("RGBA", (w, h), (10, 10, 10, 255))
draw_logo = ImageDraw.Draw(img_logo)

# Rounded rectangle container
draw_logo.rounded_rectangle((0, 0, w, h), radius=32, fill=(10, 10, 10, 255))

# Draw emblem at scale 2x
draw_infinity_emblem(draw_logo, ox=72, oy=70, scale=2.0)

try:
    font_main = ImageFont.truetype("arialbd.ttf", 76)
    font_sub = ImageFont.truetype("arialbd.ttf", 23)
except:
    font_main = ImageFont.load_default()
    font_sub = ImageFont.load_default()

draw_logo.text((232, 70), "InfinityDB", fill=(255, 255, 255, 255), font=font_main)
draw_logo.text((236, 175), "REMEMBER EVERYTHING", fill=(156, 163, 175, 255), font=font_sub)

img_logo.convert("RGB").save("assets/logo.png", "PNG")
print("[+] Created assets/logo.png (1040x280)")


# 2. Exact Square Thumbnail for Product Hunt (512x512)
sq_size = 512
img_sq = Image.new("RGBA", (sq_size, sq_size), (10, 10, 10, 255))
draw_sq = ImageDraw.Draw(img_sq)

# Draw centered emblem at 3.5x scale
emblem_ox = (sq_size - int(76 * 3.5)) // 2
emblem_oy = 85
draw_infinity_emblem(draw_sq, ox=emblem_ox, oy=emblem_oy, scale=3.5)

# Centered Typography
title = "InfinityDB"
bbox = draw_sq.textbbox((0, 0), title, font=font_main)
tw = bbox[2] - bbox[0]
draw_sq.text(((sq_size - tw) // 2, 360), title, fill=(255, 255, 255, 255), font=font_main)

subtitle = "REMEMBER EVERYTHING"
bbox_s = draw_sq.textbbox((0, 0), subtitle, font=font_sub)
sw = bbox_s[2] - bbox_s[0]
draw_sq.text(((sq_size - sw) // 2, 445), subtitle, fill=(156, 163, 175, 255), font=font_sub)

img_sq.convert("RGB").save("assets/thumbnail.png", "PNG")
print("[+] Created assets/thumbnail.png (512x512) with exact README emblem & typography")
