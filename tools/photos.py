"""Bring product photos onto the site's tile grey and centre them on a 3:4 canvas.

usage: python3 tools/photos.py SRC_DIR OUT_DIR [--gradient] [--same-size]
       (SRC_DIR holds <slug>.jpg; needs numpy, scipy, Pillow)

The backdrop is levelled to the tile grey (231, 231, 231), flattened from the frame edges inward,
and the piece is centred on an 864 x 1152 canvas so wide bags and belts are not cropped by the 3:4 tiles.
A photo whose backdrop is a grey panel inside a white frame must be cropped to the panel first.
Add --gradient for a studio backdrop that darkens towards one edge: it is levelled across rows and columns.
Add --same-size when SRC_DIR holds one piece in several colours: the body of the piece (without thin
straps, handles or laces) gets the same size and the same spot in every photo, so the colours line up.
"""
import sys, glob, os
import numpy as np
from PIL import Image
from scipy import ndimage

GREY = 231.0
W, H = 864, 1152
BOX_W, BOX_H = 0.80 * W, 0.74 * H   # the piece fits inside this box
MAX_UP = 2.4                        # never upscale more than this
MARGIN = 0.04                       # --same-size keeps everything this far from the canvas edge


def level(path, gradient=False):
    """Backdrop levelled to the tile grey; returns the image, the piece mask and the backdrop colour."""
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
    fg = ndimage.binary_opening(~back & (diff >= 9), iterations=2)
    return im, fg, bg


def full_box(fg):
    """Rows / columns with enough piece pixels: the whole piece, straps included."""
    h, w = fg.shape
    rows = np.where(fg.sum(1) > max(3, w * 0.004))[0]
    cols = np.where(fg.sum(0) > max(3, h * 0.004))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def body_box(fg):
    """The piece without thin parts: an opening wide enough to drop straps, handles and laces.
    The opening is sized on the piece itself, so a strap is dropped however large the piece sits in the frame."""
    y0, y1, x0, x1 = full_box(fg)
    r = max(3, int(min(y1 - y0, x1 - x0) * 0.08))
    # close the light spots of a pattern first, or the opening would eat into the body as well
    solid = ndimage.binary_fill_holes(ndimage.binary_closing(fg, iterations=3))
    body = ndimage.binary_opening(solid, structure=np.ones((r, r), bool))
    lab, n = ndimage.label(body)
    if n:
        body = lab == (np.argmax(ndimage.sum(body, lab, range(1, n + 1))) + 1)
    else:
        body = fg
    ys, xs = np.where(body)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def canvas(im, box, scale, at, to=(W / 2, H / 2)):
    """Crop box from im, scale it, paste it so that point `at` (in source pixels) lands on canvas point `to`."""
    y0, y1, x0, x1 = box
    piece = Image.fromarray(im[y0:y1, x0:x1].astype(np.uint8))
    piece = piece.resize((max(1, round(piece.width * scale)), max(1, round(piece.height * scale))), Image.LANCZOS)
    out = Image.new('RGB', (W, H), (231, 231, 231))
    ax, ay = at
    out.paste(piece, (round(to[0] - (ax - x0) * scale), round(to[1] - (ay - y0) * scale)))
    return out


def process(path, gradient=False):
    im, fg, bg = level(path, gradient)
    y0, y1, x0, x1 = full_box(fg)
    s = min(BOX_W / (x1 - x0), BOX_H / (y1 - y0), MAX_UP)
    return canvas(im, (y0, y1, x0, x1), s, ((x0 + x1) / 2, (y0 + y1) / 2)), bg, s


def same_size(paths, gradient=False):
    """One piece in several colours: equal body size, body centred on the same spot."""
    items = []
    for p in paths:
        im, fg, bg = level(p, gradient)
        fb, bb = full_box(fg), body_box(fg)
        size = np.sqrt((bb[1] - bb[0]) * (bb[3] - bb[2]))
        centre = ((bb[2] + bb[3]) / 2, (bb[0] + bb[1]) / 2)
        items.append((p, im, fb, size, centre, bg))
    # the size each photo would get on its own, measured on the body; keep the median
    target = float(np.median([min(BOX_W / (fb[3] - fb[2]), BOX_H / (fb[1] - fb[0])) * size
                              for _, _, fb, size, _, _ in items]))
    # reach of each photo around its body, in body units; the body sits off centre by the amount that
    # centres the widest reach of the group, so the longest strap costs the least size
    ext = [((cx - x0) / size, (x1 - cx) / size, (cy - y0) / size, (y1 - cy) / size)
           for _, _, (y0, y1, x0, x1), size, (cx, cy), _ in items]
    dx = (max(e[1] for e in ext) - max(e[0] for e in ext)) / 2
    dy = (max(e[3] for e in ext) - max(e[2] for e in ext)) / 2
    # shrink if a strap or handle would leave the canvas in any of the photos
    room_x, room_y = W / 2 - MARGIN * W, H / 2 - MARGIN * H
    for l, r, u, d in ext:
        for reach, room in ((l + dx, room_x), (r - dx, room_x), (u + dy, room_y), (d - dy, room_y)):
            if reach > 0:
                target = min(target, room / reach)
    to = (W / 2 - dx * target, H / 2 - dy * target)
    out = []
    for p, im, fb, size, centre, bg in items:
        s = min(target / size, MAX_UP)
        out.append((p, canvas(im, fb, s, centre, to), bg, s))
    return out


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    gradient = '--gradient' in sys.argv[3:]
    os.makedirs(dst, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(src, '*.jpg')))
    if '--same-size' in sys.argv[3:]:
        results = same_size(paths, gradient)
    else:
        results = [(p, *process(p, gradient)) for p in paths]
    for p, out, bg, s in results:
        name = os.path.splitext(os.path.basename(p))[0]
        out.save(os.path.join(dst, name + '.webp'), 'WEBP', quality=84, method=6)
        print(f'{name:32s} bg {bg.round().astype(int)} scale {s:.2f}')
