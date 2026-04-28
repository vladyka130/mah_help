"""Одноразовий діагностичний скрипт: макс. score matchTemplate по всіх шаблонах."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    image_path = Path("game_frame_mahjong.png")
    t_dir = Path("assets/tiles")
    frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print("Немає кадру:", image_path, file=sys.stderr)
        raise SystemExit(1)

    best: list[tuple[float, str, tuple[int, int], tuple[int, int]]] = []
    for p in sorted(t_dir.glob("*.png")):
        t = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if t is None or t.size == 0:
            continue
        if t.shape[0] > frame.shape[0] or t.shape[1] > frame.shape[1]:
            continue
        r = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
        mx = float(r.max())
        yx = np.unravel_index(int(r.argmax()), r.shape)
        best.append((mx, p.name, (int(yx[0]), int(yx[1])), (t.shape[0], t.shape[1])))

    best.sort(key=lambda x: -x[0])
    print("Кадр:", image_path.resolve(), "shape", frame.shape)
    print("Топ-15 макс. score:")
    for row in best[:15]:
        print(
            f"  {row[1]:20s} max={row[0]:.4f} pos=({row[2][0]},{row[2][1]}) template={row[3]}"
        )
    if best:
        print("Найгірший з top:", best[-1] if len(best) < 15 else "…")


if __name__ == "__main__":
    main()
