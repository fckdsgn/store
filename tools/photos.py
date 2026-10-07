"""Bring product photos onto the site's tile grey and centre them on a 3:4 canvas.

usage: python3 tools/photos.py SRC_DIR OUT_DIR   (SRC_DIR holds <slug>.jpg; needs numpy, scipy, Pillow)

The backdrop is levelled to the tile grey (231, 231, 231), flattened from the frame edges inward,
and the piece is centred on an 864 x 1152 canvas so wide bags and belts are not cropped by the 3:4 tiles.
A photo whose backdrop is a grey panel inside a white frame must be cropped to the panel first.
Add --gradient for a studio backdrop that darkens towards one edge: it is levelled across rows and columns.
"""
import sys, glob, os
import numpy as np
from PIL import Image
from scipy import ndimage

GREY = 231.0
W, H = 864, 1152
BOX_W, BOX_H = 0.80 * W, 0.74 * H   # the piece fits inside this box
MAX_UP = 2.4                        # never upscale more than this

def process(path, gradient=False):
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    h, w, _ = im.shape
    b = max(2, int(min(h, w) * 0.02))
    if gradient:
        # backdrop model from the frame: rows from the left/right edges, columns from the top/bottom edges
        row_bg = ndimage.median_filter(np.median(np.concatenate([im[:, :b], im[:, -b:]], axis=1), axis=1), size=(9, 1))
        col_bg = ndimage.median_filter(np.median(np.concatenate([im[:b], im[-b:]], axis=0), axis=0), size=(9, 1))
        plate = row_bg[:, None, :] * col_bg[None, :, :] / np.maximum(np.median(row_bg, axis=0), 1)
        im = np.clip(im * (GREY / np.maximum(plate, 1)), 0, 255)
    border = np.concatenate([im[:b].reshape(-1, 3), im[-b:].reshape(-1, 3),
                             im[:, :b].reshape(-1, 3), im[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    # level brightness so the backdrop lands on the tile grey (keeps soft edges and shadows)
    im = np.clip(im * (GREY / np.maximum(bg, 1)), 0, 255)
    # backdrop = everything close to grey that touches the frame edge
    diff = np.abs(im - GREY).max(axis=2)
    near = diff < 9
    lab, _ = ndimage.label(near)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    back = np.isin(lab, edge[edge > 0])
    # soften the seam between flattened backdrop and the rest
    soft = ndimage.gaussian_filter(back.astype(np.float32), 1.2)[..., None]
    im = im * (1 - soft) + GREY * soft
    # bounding box of the piece: rows / columns with enough non-backdrop pixels
    fg = ~back & (diff >= 9)
    fg = ndimage.binary_opening(fg, iterations=2)
    rows = np.where(fg.sum(1) > max(3, w * 0.004))[0]
    cols = np.where(fg.sum(0) > max(3, h * 0.004))[0]
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    piece = Image.fromarray(im[y0:y1, x0:x1].astype(np.uint8))
    s = min(BOX_W / piece.width, BOX_H / piece.height, MAX_UP)
    piece = piece.resize((round(piece.width * s), round(piece.height * s)), Image.LANCZOS)
    out = Image.new('RGB', (W, H), (231, 231, 231))
    out.paste(piece, ((W - piece.width) // 2, (H - piece.height) // 2))
    return out, bg, s

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    gradient = '--gradient' in sys.argv[3:]
    os.makedirs(dst, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src, '*.jpg'))):
        out, bg, s = process(p, gradient)
        name = os.path.splitext(os.path.basename(p))[0]
        out.save(os.path.join(dst, name + '.webp'), 'WEBP', quality=84, method=6)
        print(f'{name:32s} bg {bg.round().astype(int)} scale {s:.2f}')
