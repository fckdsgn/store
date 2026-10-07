"""Click areas on the home video: the pieces she wears link to their product pages.

usage: python3 tools/hero-hotspots.py assets/hero/hero-1920.mp4 [CHECK.png]   (needs ffmpeg, numpy, opencv-python-headless)

The hoodie (pink) and the bag (brown monogram and tan leather) are found by colour in every other frame of the loop and
joined, so the area covers them in every pose (she lifts her arm to her head mid-clip). The snow mask and the sneakers
are traced by hand (black goggles on black hair and white sneakers on a white floor do not separate by colour).
Prints the HOT list for V.home in index.html: SVG paths in frame pixels (1920 x 1080). With CHECK.png it also draws
the areas over a few frames, to look at.
"""
import subprocess, sys
import numpy as np
import cv2

W, H = 1920, 1080
GOGGLES = [(990, 110), (1060, 88), (1200, 85), (1258, 110), (1262, 185), (1240, 222), (1000, 226), (985, 185)]
LEFT_SHOE = [(612, 1003), (650, 982), (700, 955), (740, 930), (742, 850), (760, 825), (800, 808), (870, 808),
             (925, 820), (940, 850), (943, 1008), (925, 1032), (860, 1040), (780, 1049), (700, 1047), (640, 1036),
             (612, 1018)]
RIGHT_SHOE = [(1000, 880), (1040, 858), (1110, 850), (1180, 870), (1250, 890), (1318, 920), (1314, 965), (1298, 1010),
              (1270, 1046), (1244, 1054), (1221, 1053), (1206, 1050), (1192, 1030), (1186, 1007), (1150, 983),
              (1080, 967), (1010, 960), (995, 930)]

raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', sys.argv[1], '-vf', "select='not(mod(n\\,2))'", '-vsync', '0',
                      '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], check=True, capture_output=True).stdout
frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
pink = np.zeros((H, W), np.uint8)
bag = np.zeros((H, W), np.uint8)
box = np.zeros((H, W), np.uint8)
box[480:930, 280:900] = 1                                   # where the bag sits
for f in frames:
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    p = ((h >= 150) & (s > 45) & (v > 110)).astype(np.uint8)
    pink |= cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    b = (((h >= 3) & (h <= 22) & (v < 120) & (s > 40)) | ((h >= 8) & (h <= 20) & (s > 115) & (v >= 120) & (v <= 230)))
    bag |= cv2.morphologyEx(b.astype(np.uint8) & box, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def outline(m, close):
    """The largest piece of the mask, holes filled, as a simplified polygon."""
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    return [tuple(p) for p in cv2.approxPolyDP(c, 5, True)[:, 0]]


areas = [('balenciaga-soccer-hoodie', 'pink', outline(pink, 25)), ('lv-keepall-55', 'monogram', outline(bag, 31)),
         ('margiela-future-high', '', LEFT_SHOE), ('margiela-future-high', '', RIGHT_SHOE),
         ('lv-snow-mask', '', GOGGLES)]                       # later areas lie on top: the mask over the hoodie
print('const HOT = [')
for pid, v, poly in areas:
    d = 'M' + 'L'.join(f'{x} {y}' for x, y in poly) + 'Z'
    print(f"    ['{pid}', '{v}', '{d}'],")
print('  ];')

if len(sys.argv) > 2:
    tiles = []
    for f in frames[::len(frames) // 6][:6]:
        im = f.copy()
        for _, _, poly in areas:
            cv2.polylines(im, [np.array(poly, np.int32)], True, (255, 0, 255), 3)
        tiles.append(cv2.resize(im[40:, 250:1650], (700, 520)))
    cv2.imwrite(sys.argv[2], np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
