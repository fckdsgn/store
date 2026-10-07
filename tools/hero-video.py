"""Home page video: an endless loop, toned to the tile grey, with the sneakers stepping over the plate's bottom edge.

usage: python3 tools/hero-video.py assets/hero/hero-source.mp4 assets/hero   (needs ffmpeg, numpy, Pillow)

Makes, in OUT:
  hero-VER-2560.mp4   wide screens: 2560 px, lightly denoised and sharpened, so Retina screens get a crisp picture
  hero-VER-1920.mp4   the 1920 px master the smaller files are made from
  hero-VER-1600.mp4   phones and tablets: 2.6 MB instead of 7, and on a phone screen it looks the same as 1920
  hero-VER-1920.webm  for browsers without H.264 (Chromium builds, some Linux Firefox)
  hero-VER-poster.jpg the first frame, shown until the video plays
  (VER changes with every new cut: browsers and the preview keep old files cached under the same name)
  hero-shoes.png  the mask that lets the sneakers show below the plate (index.html: .reel video mask)

The loop: the clip plays forward and then backward, so its last frame leads straight back into its first one (the
owner turned down a dissolve, and a plain cut jumps: the camera drifts and never returns to its first frame).
FROM / TO are the turnaround frames, at moments where the model barely moves.

The tone: the white studio backdrop is multiplied down to the tile grey (231). Below LINE (the plate's bottom edge)
the floor is lifted to pure white instead, the page colour, while the sneakers keep the plate's tone: they are
darker than the floor (about 229 against 240 to 250), so a curve on brightness separates them. The video is then
masked to the plate plus the sneakers' outline (with room for their contact shadows); since the floor around them is
white like the page, the mask edge does not show, and the sneakers and their real shadows step out of the plate.
White sneakers on a white floor defeat automatic cut-out, so their outlines are traced by hand on a zoomed frame
(source pixels, 1920 x 1080). A new video needs new outlines, LINE, FROM / TO and new tag spots in index.html.
"""
import os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SRC, OUT = sys.argv[1], sys.argv[2]
VER = 'v3'
W, H = 1920, 1080
FROM, TO = 12, 230          # turnaround frames
TONE = 0.93                 # backdrop ~248 * 0.93 = the tile grey 231
LINE = 962                  # plate bottom edge, source row (89% of 1080): under the bag, the hand and the legs
DARK, WHITE = 222, 240      # below LINE: brightness up to DARK keeps the plate tone (sneakers), from WHITE on is page white
LEFT_SHOE = [(706, 958), (697, 962), (650, 982), (630, 991), (618, 998), (616, 1005), (618, 1013), (626, 1021),
             (640, 1028), (650, 1033), (675, 1042), (700, 1047), (725, 1049), (750, 1048), (775, 1045), (800, 1041),
             (825, 1036), (850, 1032), (875, 1028), (900, 1025), (925, 1020), (936, 1015), (941, 1008), (942, 1000),
             (941, 958)]
RIGHT_SHOE = [(1062, 958), (1308, 958), (1306, 962), (1300, 973), (1292, 982), (1283, 992), (1277, 1002),
              (1270, 1010), (1258, 1022), (1250, 1030), (1238, 1040), (1230, 1048), (1221, 1053), (1206, 1050),
              (1198, 1042), (1192, 1030), (1188, 1018), (1186, 1007), (1172, 992), (1150, 983), (1130, 977),
              (1100, 971), (1080, 967)]


def outline(polys, grow=0, shift=(0,), blur=0.0):
    """The sneakers' outlines as a smooth 0..1 map, source size."""
    ss = 4
    big = Image.new('L', (W * ss, H * ss), 0)
    d = ImageDraw.Draw(big)
    for poly in polys:
        for dy in shift:
            d.polygon([(x * ss, (y + dy) * ss) for x, y in poly], fill=255)
    m = big.resize((W, H), Image.LANCZOS)
    if grow > 0: m = m.filter(ImageFilter.MaxFilter(2 * grow + 1))
    if grow < 0: m = m.filter(ImageFilter.MinFilter(-2 * grow + 1))
    if blur: m = m.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(m, np.float32) / 255


# inside the sneakers (a hair inside their outline) the plate tone carries on below the line, so they do not change
# brightness at the plate's edge; outside them only the floor is lifted to white, darker shadows stay
KEEP = outline((LEFT_SHOE, RIGHT_SHOE), grow=-2, blur=1.5)[LINE:, :, None]


def tone(frame):
    v = frame.astype(np.float32)
    out = v * TONE
    low = v[LINE:]
    lum = low.mean(axis=2, keepdims=True)
    t = np.clip((lum - DARK) / (WHITE - DARK), 0, 1)
    t = t * t * (3 - 2 * t)                                   # smoothstep
    lifted = low * ((lum * TONE * (1 - t) + 255 * t) / np.maximum(lum, 1))
    out[LINE:] = KEEP * out[LINE:] + (1 - KEEP) * lifted
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def shoes_mask():
    """Alpha mask of the sneakers below the line, cropped; returns it and its box in source pixels."""
    # a little past the outline and down, for the contact shadows; the floor there is white, so the edge does not show
    a = (outline((LEFT_SHOE, RIGHT_SHOE), grow=4, shift=(0, 6, 12), blur=4) * 255 + 0.5).astype(np.uint8)
    top = LINE - 16                                           # overlaps the plate, so the blur never leaves a gap
    a[:top] = 0
    ys, xs = np.where(a > 0)
    x0, x1, y1 = xs.min(), xs.max() + 1, min(H, ys.max() + 1)
    crop = a[top:y1, x0:x1]
    rgba = np.zeros(crop.shape + (4,), np.uint8)
    rgba[..., 3] = crop
    return Image.fromarray(rgba), (x0, top, x1, y1)


# read the loop's frames
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', SRC, '-vf', f'trim=start_frame={FROM}:end_frame={TO + 1},setpts=PTS-STARTPTS',
                      '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
order = list(range(len(frames))) + list(range(len(frames) - 2, 0, -1))   # forward, then back without repeating ends

hd = 'hqdn3d=1:1:0:0,scale=2560:1440:flags=lanczos,unsharp=5:5:0.7:5:5:0'
sd = 'hqdn3d=1:1:0:0,unsharp=5:5:0.5:5:5:0'
x264 = ['-c:v', 'libx264', '-preset', 'slow', '-tune', 'film', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart']
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                        '-r', '24', '-i', '-', '-filter_complex', f'[0:v]split=2[a][b];[a]{hd}[hd];[b]{sd}[sd]',
                        '-map', '[hd]', *x264, '-crf', '19', os.path.join(OUT, f'hero-{VER}-2560.mp4'),
                        '-map', '[sd]', *x264, '-crf', '19', os.path.join(OUT, f'hero-{VER}-1920.mp4')],
                       stdin=subprocess.PIPE)
for i in order:
    enc.stdin.write(tone(frames[i]).tobytes())
enc.stdin.close()
assert enc.wait() == 0
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(OUT, f'hero-{VER}-1920.mp4'), '-c:v', 'libvpx-vp9',
                '-b:v', '0', '-crf', '26', '-row-mt', '1', '-deadline', 'good', '-cpu-used', '1',
                os.path.join(OUT, f'hero-{VER}-1920.webm')], check=True)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(OUT, f'hero-{VER}-1920.mp4'), '-vf', 'scale=1600:-2:flags=lanczos',
                *x264, '-crf', '24', os.path.join(OUT, f'hero-{VER}-1600.mp4')], check=True)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(OUT, f'hero-{VER}-2560.mp4'), '-frames:v', '1',
                '-q:v', '2', os.path.join(OUT, f'hero-{VER}-poster.jpg')], check=True)

mask, (x0, y0, x1, y1) = shoes_mask()
mask.save(os.path.join(OUT, 'hero-shoes.png'), optimize=True)
print(f'mask {x1 - x0} x {y1 - y0}: left {x0 / W:.5f}, top {y0 / H:.5f}, width {(x1 - x0) / W:.5f} of the frame; '
      f'plate line {LINE}/{H}')
