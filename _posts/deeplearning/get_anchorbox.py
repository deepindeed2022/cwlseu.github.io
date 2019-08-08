# import torch
import numpy as np

def get_anchors_over_grid(ratios, scales, stride):
    """
    generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    ratios = np.array(ratios)
    scales = np.array(scales)
    anchor = np.array([1, 1, stride, stride], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, ratios)
    anchors = np.vstack([_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])])
    return anchors


def _ratio_enum(anchor, ratios):
    """enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _whctrs(anchor):
    """return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


if __name__ == '__main__':
    ratios = [0.5, 1, 2]
    scales = [3, 6, 12, 22]
    stride = 16
    anchors = get_anchors_over_grid(ratios, scales, stride)
    print(anchors)
    h, w = anchors.shape
    print("["),
    for i in xrange(h):
    	print("["),
    	for j in xrange(w - 1):
    		print("{},".format(anchors[i][j])),
    	print("{}],".format(anchors[i][w-1]))
    print("]")