"""Cut the sneakers out of one frame of the home video, for the part that steps over the bottom edge of the plate.

usage: python3 tools/hero-shoes.py FRAME.png OUT.webp   (needs numpy, Pillow)

FRAME.png is one frame of the source video, toned the same way as the video (tools/hero-video.sh makes it); the
sneakers do not move in the owner's video, so one frame matches the whole clip. White sneakers on a white floor
defeat automatic cut-out, so the outlines below are traced by hand on a zoomed frame (source pixels, 1920 x 1080)
and drawn with smooth edges. A new video needs new outlines.
OUT.webp holds the sneakers below LINE (the plate's bottom edge), cropped to them; the script prints where the
cut-out sits in the frame, in % of the frame, for the .pop img rule in index.html.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LINE = 962   # plate bottom edge, in source rows (89% of 1080): under the bag, the hand and the legs
LEFT_SHOE = [(706, 958), (697, 962), (650, 982), (630, 991), (618, 998), (616, 1005), (618, 1013), (626, 1021), (640, 1028),
             (650, 1033), (675, 1042), (700, 1047), (725, 1049), (750, 1048), (775, 1045), (800, 1041), (825, 1036),
             (850, 1032), (875, 1028), (900, 1025), (925, 1020), (936, 1015), (941, 1008), (942, 1000), (941, 958)]
RIGHT_SHOE = [(1062, 958), (1308, 958), (1306, 962), (1300, 973), (1292, 982), (1283, 992), (1277, 1002), (1270, 1010),
              (1258, 1022), (1250, 1030), (1238, 1040), (1230, 1048), (1221, 1053), (1206, 1050), (1198, 1042),
              (1192, 1030), (1188, 1018), (1186, 1007), (1172, 992), (1150, 983), (1130, 977), (1100, 971), (1080, 967)]
SS = 4       # draw the outlines this many times larger, then shrink: smooth, anti-aliased edges
OVER = 4     # the cut-out starts this many rows above the line, so no gap shows at the plate's edge

im = Image.open(sys.argv[1]).convert('RGB')
w, h = im.size
big = Image.new('L', (w * SS, h * SS), 0)
d = ImageDraw.Draw(big)
for poly in (LEFT_SHOE, RIGHT_SHOE):
    d.polygon([(x * SS, y * SS) for x, y in poly], fill=255)
alpha = big.resize((w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.5))
a = np.asarray(alpha).copy()
a[:LINE - OVER] = 0          # above the line the video itself shows the sneakers
out = np.dstack([np.asarray(im), a])
ys, xs = np.where(a > 0)
x0, x1, y1 = xs.min(), xs.max() + 1, ys.max() + 1
Image.fromarray(out[LINE - OVER:y1, x0:x1]).save(sys.argv[2], 'WEBP', quality=92, alpha_quality=100, method=6)
print('cut-out %d x %d: left %.3f%%; top %.3f%%; width %.3f%% (plate line %d / %d = %.4f)'
      % (x1 - x0, y1 - LINE + OVER, x0 / w * 100, (LINE - OVER) / h * 100, (x1 - x0) / w * 100, LINE, h, LINE / h))
