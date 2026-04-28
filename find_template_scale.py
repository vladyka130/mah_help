"""Підбір масштабу шаблону до кадру (грубий пошук)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    frame = cv2.imread("game_frame_mahjong.png", cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print("Немає game_frame_mahjong.png", file=sys.stderr)
        raise SystemExit(1)
    tpl_path = Path("assets/tiles/1.png")
    template = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("Немає шаблону", tpl_path, file=sys.stderr)
        raise SystemExit(1)

    best: tuple[float, float, tuple[int, int], tuple[int, int]] = (
        -1.0,
        0.0,
        (0, 0),
        (0, 0),
    )
    for scale in np.linspace(0.25, 1.5, 51):
        h = int(template.shape[0] * scale)
        w = int(template.shape[1] * scale)
        if h < 20 or w < 20 or h > frame.shape[0] or w > frame.shape[1]:
            continue
        t2 = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(frame, t2, cv2.TM_CCOEFF_NORMED)
        mx = float(r.max())
        yx = np.unravel_index(int(r.argmax()), r.shape)
        if mx > best[0]:
            best = (mx, float(scale), (int(yx[0]), int(yx[1])), (h, w))

    print("Кадр", frame.shape, "шаблон 1.png ориг.", template.shape)
    print("Найкращий scale:", best[1], "max score:", best[0], "pos", best[2], "resized", best[3])


if __name__ == "__main__":
    main()
