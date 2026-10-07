"""Click areas and press outlines for the clothes on the home video.

usage: python3 tools/hero-hotspots.py assets/hero/hero-v3-1920.mp4 assets/hero/hero-outlines.json [CHECK.png]
       (needs ffmpeg, numpy, opencv-python-headless)

The loop plays source frames FROM..TO forward and then back (tools/hero-video.py), so its first half holds every pose.

Click areas (printed as the HOT list for V.home in index.html, SVG paths in frame pixels, 1920 x 1080): the hoodie (pink)
and the bag (brown monogram, tan leather) are found by colour in every frame and joined, so they hold in every pose;
the snow mask is the union of its traced outlines, the sneakers are their traced outlines.

Press outlines (written to the JSON file, loaded by the page): the grey line drawn along the piece while it is pressed.
The hoodie is outlined by colour in every other frame, the mask moves with her head, so it is traced by hand on a
keyframe every second and blended in between (black goggles on black hair do not separate by colour); the bag and the
sneakers stand still and have one outline each. Paths are relative ('l' steps), which keeps the file small.
With CHECK.png the outlines are drawn over a few frames, to look at.
"""
import json, subprocess, sys
import numpy as np
import cv2

W, H = 1920, 1080
FROM = 12                                  # first source frame of the loop (tools/hero-video.py)
STEP = 2                                   # hoodie and mask outlines for every other frame
# the snow mask, traced on source frames (keys), the same 12 points in the same order each time
GOGGLES = {
    12:  [(1062, 150), (1067, 193), (1093, 200), (1127, 207), (1152, 200), (1163, 167), (1200, 188), (1227, 177), (1223, 133), (1200, 117), (1157, 112), (1103, 132)],
    24:  [(1057, 148), (1060, 193), (1093, 202), (1137, 212), (1153, 200), (1162, 175), (1200, 197), (1223, 187), (1225, 143), (1203, 127), (1160, 122), (1103, 132)],
    48:  [(1025, 160), (1032, 210), (1057, 220), (1087, 225), (1097, 213), (1102, 188), (1160, 217), (1215, 203), (1210, 160), (1193, 142), (1133, 137), (1060, 143)],
    72:  [(1037, 123), (1032, 160), (1060, 173), (1092, 160), (1103, 145), (1117, 140), (1173, 182), (1198, 175), (1200, 132), (1177, 107), (1110, 97), (1060, 107)],
    96:  [(1040, 118), (1032, 158), (1057, 167), (1087, 150), (1097, 137), (1110, 132), (1167, 160), (1195, 155), (1193, 107), (1167, 93), (1110, 90), (1060, 98)],
    120: [(1055, 122), (1048, 187), (1070, 192), (1093, 170), (1100, 150), (1112, 147), (1160, 175), (1198, 160), (1198, 118), (1173, 102), (1117, 100), (1073, 110)],
    144: [(1057, 123), (1048, 187), (1070, 193), (1093, 170), (1100, 152), (1112, 148), (1160, 177), (1200, 170), (1207, 122), (1180, 105), (1120, 102), (1073, 112)],
    168: [(1075, 110), (1067, 155), (1093, 160), (1117, 143), (1130, 132), (1142, 130), (1193, 170), (1230, 158), (1227, 117), (1200, 97), (1147, 88), (1100, 95)],
    192: [(1073, 127), (1075, 167), (1100, 177), (1133, 175), (1157, 167), (1167, 148), (1203, 173), (1233, 163), (1230, 127), (1207, 103), (1153, 93), (1100, 105)],
    216: [(1057, 142), (1057, 187), (1087, 197), (1127, 200), (1148, 190), (1157, 165), (1193, 190), (1223, 180), (1222, 137), (1200, 118), (1153, 108), (1100, 125)],
}
GOGGLES_RIM = 7                            # the points sit on the lenses; the black frame reaches this much further
LEFT_SHOE = [(617, 1003), (630, 987), (657, 972), (690, 960), (700, 953), (730, 930), (757, 910), (770, 900), (765, 893),
             (777, 870), (790, 847), (803, 820), (810, 813), (830, 812), (857, 823), (890, 837), (920, 853), (933, 860),
             (933, 890), (935, 930), (938, 967), (940, 997), (941, 1008), (936, 1015), (925, 1020), (900, 1025),
             (875, 1028), (850, 1032), (825, 1036), (800, 1041), (775, 1045), (750, 1048), (725, 1049), (700, 1047),
             (675, 1042), (650, 1033), (640, 1028), (626, 1021), (618, 1013)]
RIGHT_SHOE = [(1060, 913), (1063, 900), (1077, 877), (1090, 865), (1107, 860), (1130, 858), (1153, 863), (1180, 873),
              (1200, 882), (1213, 888), (1247, 892), (1280, 898), (1300, 902), (1313, 910), (1315, 923), (1312, 940),
              (1308, 955), (1306, 962), (1300, 973), (1292, 982), (1283, 992), (1277, 1002), (1270, 1010), (1258, 1022),
              (1250, 1030), (1238, 1040), (1230, 1048), (1221, 1053), (1206, 1050), (1198, 1042), (1192, 1030),
              (1188, 1018), (1186, 1007), (1172, 992), (1150, 983), (1130, 977), (1100, 971), (1080, 967), (1067, 957),
              (1062, 940)]

raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', sys.argv[1], '-vf', 'trim=end_frame=219', '-f', 'rawvideo',
                      '-pix_fmt', 'bgr24', '-'], check=True, capture_output=True).stdout
frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)          # the forward half: every pose of the loop


def goggles_at(j):
    """The mask's outline on loop frame j, blended between the nearest traced keys."""
    s, ks = FROM + j, sorted(GOGGLES)
    s = min(max(s, ks[0]), ks[-1])
    for a, b in zip(ks, ks[1:]):
        if a <= s <= b:
            t = (s - a) / (b - a)
            return np.round((1 - t) * np.array(GOGGLES[a]) + t * np.array(GOGGLES[b])).astype(np.int32)


def colour_masks(f):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    pink = cv2.morphologyEx(((h >= 150) & (s > 45) & (v > 110)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bag = (((h >= 3) & (h <= 22) & (v < 120) & (s > 40)) | ((h >= 8) & (h <= 20) & (s > 115) & (v >= 120) & (v <= 230)))
    bag = bag.astype(np.uint8)
    bag[:480] = 0; bag[930:] = 0; bag[:, :280] = 0; bag[:, 900:] = 0                   # where the bag sits
    return pink, cv2.morphologyEx(bag, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def fill(poly, grow=0):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)) if grow else m


def smooth(m, close, blur=9):
    """Bridge hair strands that cross the piece and round off the pixel steps, so the outline runs smoothly."""
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    return (cv2.GaussianBlur(m.astype(np.float32), (0, 0), blur / 3) > 0.5).astype(np.uint8)


def contours(m, eps, min_area, close=0):
    if close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [cv2.approxPolyDP(c, eps, True)[:, 0].tolist() for c in cs if cv2.contourArea(c) >= min_area]


def path(cs, relative=True):
    out = ''
    for c in cs:
        if relative:
            steps, (x0, y0) = [], c[0]
            for x, y in c[1:]:
                steps.append(f'{x - x0} {y - y0}')
                x0, y0 = x, y
            out += f'M{c[0][0]} {c[0][1]}l' + ' '.join(steps) + 'z'
        else:
            out += 'M' + 'L'.join(f'{x} {y}' for x, y in c) + 'Z'
    return out.replace(' -', '-')


pink_all, gog_all = (np.zeros((H, W), np.uint8) for _ in range(2))
bag_count = np.zeros((H, W), np.uint16)
hood_lines, gog_lines = [], []
for j, f in enumerate(frames):
    pink, bag = colour_masks(f)
    pink_all |= pink
    bag_count += bag
    gog = fill(goggles_at(j), GOGGLES_RIM)
    gog_all |= gog
    if j % STEP == 0:
        hood_lines.append(path(contours(smooth(pink, 21), 3, 2500)))
        gog_lines.append(path(contours(gog, 2, 100)))
# the bag stands still: keep what is bag in most frames, so her hand passing by (skin like tan leather) drops out
bag_all = (bag_count >= 0.6 * len(frames)).astype(np.uint8)
bag_line = path(sorted(contours(smooth(bag_all, 15), 2.5, 5000), key=lambda c: -cv2.contourArea(np.array(c, np.int32)))[:1])
shoes_line = path([LEFT_SHOE, RIGHT_SHOE])

json.dump({'step': STEP, 'half': len(frames), 'hood': hood_lines, 'gog': gog_lines, 'bag': bag_line, 'shoes': shoes_line},
          open(sys.argv[2], 'w'), separators=(',', ':'))

biggest = lambda cs: [max(cs, key=lambda c: cv2.contourArea(np.array(c, np.int32)))]
areas = [('balenciaga-soccer-hoodie', 'pink', 'hood', biggest(contours(pink_all, 5, 1000, close=25))),
         ('lv-keepall-55', 'monogram', 'bag', biggest(contours(bag_all, 5, 1000, close=31))),
         ('margiela-future-high', '', 'shoes', contours(fill(LEFT_SHOE, 4) | fill(RIGHT_SHOE, 4), 3, 1000)),
         ('lv-snow-mask', '', 'gog', contours(gog_all, 3, 1000))]   # later areas lie on top: the mask over the hoodie
print('const HOT = [')
for pid, v, key, cs in areas:
    print(f"    ['{pid}', '{v}', '{key}', '{path(cs, relative=False)}'],")
print('  ];')

if len(sys.argv) > 3:
    tiles = []
    for j in range(0, len(frames), len(frames) // 6)[:6]:
        im = frames[j].copy()
        k = j // STEP
        for d in (hood_lines[k], gog_lines[k], bag_line, shoes_line):
            for sub in d.split('z'):
                if not sub:
                    continue
                head, steps = sub[1:].split('l')
                x, y = map(int, head.split())
                pts = [(x, y)]
                nums = [int(n) for n in steps.replace('-', ' -').split()]
                for dx, dy in zip(nums[::2], nums[1::2]):
                    x, y = x + dx, y + dy
                    pts.append((x, y))
                cv2.polylines(im, [np.array(pts, np.int32)], True, (110, 110, 110), 3)
        tiles.append(cv2.resize(im[40:, 250:1650], (700, 520)))
    cv2.imwrite(sys.argv[3], np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
