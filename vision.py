from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np


@dataclass
class TileMatch:
    """Опис знайденої плитки на екрані."""

    tile_type: str
    confidence: float
    x: int
    y: int
    w: int
    h: int


class VisionEngine:
    """Захват екрана та пошук плиток за шаблонами."""

    def __init__(
        self,
        templates_dir: str = "assets/tiles",
        threshold: float = 0.88,
        template_scale: float = 1.0,
        auto_scale: bool = True,
        auto_scale_range: Tuple[float, float] = (0.60, 1.40),
        auto_scale_step: float = 0.05,
        # Швидкий підбір масштабу (грубий крок + уточнення; прев’ю кадр; повторне використання попереднього s).
        auto_scale_coarse_step: float = 0.09,
        auto_scale_fine_step: float = 0.03,
        auto_scale_fine_half_span: float = 0.15,
        auto_scale_sample_templates: int = 6,
        auto_scale_preview_max_side: int = 1280,
        auto_scale_reuse: bool = True,
        auto_scale_reuse_half_span: float = 0.10,
        auto_scale_reuse_min_avg: float = 0.36,
    ) -> None:
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        # Множник розміру шаблону відносно оригінального файлу (калібрування під різні DPI/масштаб вікна).
        self.template_scale = float(template_scale)
        self.auto_scale = bool(auto_scale)
        self.auto_scale_range = auto_scale_range
        self.auto_scale_step = float(auto_scale_step)
        self.auto_scale_coarse_step = float(auto_scale_coarse_step)
        self.auto_scale_fine_step = float(auto_scale_fine_step)
        self.auto_scale_fine_half_span = float(auto_scale_fine_half_span)
        self.auto_scale_sample_templates = max(3, int(auto_scale_sample_templates))
        self.auto_scale_preview_max_side = int(auto_scale_preview_max_side)
        self.auto_scale_reuse = bool(auto_scale_reuse)
        self.auto_scale_reuse_half_span = float(auto_scale_reuse_half_span)
        self.auto_scale_reuse_min_avg = float(auto_scale_reuse_min_avg)
        self._last_auto_scale: Optional[float] = None
        self.templates: Dict[str, np.ndarray] = {}
        self.load_templates()

    def load_templates(self) -> None:
        """Завантажує всі PNG/JPG шаблони плиток у відтінках сірого."""
        self.templates.clear()
        if not self.templates_dir.exists():
            raise FileNotFoundError(
                f"Папку шаблонів не знайдено: {self.templates_dir.resolve()}"
            )

        supported = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        for file_path in sorted(self.templates_dir.iterdir()):
            if file_path.suffix.lower() not in supported:
                continue
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.size == 0:
                continue
            if self.template_scale != 1.0 and self.template_scale > 0:
                nh = max(1, int(round(image.shape[0] * self.template_scale)))
                nw = max(1, int(round(image.shape[1] * self.template_scale)))
                interp = cv2.INTER_AREA if self.template_scale < 1.0 else cv2.INTER_CUBIC
                image = cv2.resize(image, (nw, nh), interpolation=interp)
            self.templates[file_path.stem] = image

        if not self.templates:
            raise ValueError(
                f"У папці {self.templates_dir.resolve()} не знайдено валідних шаблонів плиток."
            )

    def capture_screen(
        self, region: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Робить швидкий скріншот через mss.

        region: (left, top, width, height) або None для всього основного монітора.

        Повертає (кадр, (left, top, width, height)) у віртуальних координатах екрана Windows.
        Координати matchTemplate — відносно кадру; для оверлею їх треба зсунути на +left,+top.
        """
        with mss.mss() as sct:
            monitor = (
                {
                    "left": int(region[0]),
                    "top": int(region[1]),
                    "width": int(region[2]),
                    "height": int(region[3]),
                }
                if region
                else sct.monitors[1]
            )
            shot = np.array(sct.grab(monitor))

        # mss повертає BGRA, конвертуємо в BGR для OpenCV.
        frame_bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        cap_rect = (
            int(monitor["left"]),
            int(monitor["top"]),
            int(monitor["width"]),
            int(monitor["height"]),
        )
        return frame_bgr, cap_rect

    def find_templates(
        self, frame_bgr: np.ndarray
    ) -> Tuple[List[TileMatch], Dict[str, List[TileMatch]]]:
        """Шукає всі шаблони на кадрі через cv2.matchTemplate."""
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scale = self.template_scale
        if self.auto_scale:
            scale = self._estimate_best_scale(frame_gray)
            self._last_auto_scale = scale

        all_matches: List[TileMatch] = []

        for tile_type, template in self.templates.items():
            scaled_template = self._scale_template(template, scale)
            h, w = scaled_template.shape[:2]
            if h < 5 or w < 5 or h > frame_gray.shape[0] or w > frame_gray.shape[1]:
                continue
            result = cv2.matchTemplate(frame_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= self.threshold)

            tile_matches: List[TileMatch] = []
            for x, y in zip(xs, ys):
                score = float(result[y, x])
                tile_matches.append(
                    TileMatch(
                        tile_type=tile_type,
                        confidence=score,
                        x=int(x),
                        y=int(y),
                        w=int(w),
                        h=int(h),
                    )
                )

            # Відсікаємо дублікати, які виникають поруч у matchTemplate.
            tile_matches = self._non_max_suppression(tile_matches, iou_threshold=0.30)
            all_matches.extend(tile_matches)

        # Глобальний NMS між усіма класами: прибирає рамки, що сильно накладаються.
        global_nms = self._non_max_suppression(all_matches, iou_threshold=0.35)
        # Winner-takes-all: для кожної фізичної плитки залишаємо лише мітку з найвищим score.
        final_matches = self._winner_takes_all(global_nms, center_radius_ratio=0.35)
        grouped = self._group_by_type(final_matches)
        return final_matches, grouped

    def _gray_preview_for_scale_search(
        self, frame_gray: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Зменшений сірий кадр лише для підбору масштабу (matchTemplate швидший).
        Повертає (кадр, множник: ефективний scale шаблону множити на нього для цього кадру).
        """
        mh, mw = frame_gray.shape[:2]
        side = max(mh, mw)
        lim = self.auto_scale_preview_max_side
        if lim <= 0 or side <= lim:
            return frame_gray, 1.0
        f = lim / float(side)
        nw = max(320, int(round(mw * f)))
        nh = max(240, int(round(mh * f)))
        small = cv2.resize(frame_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        return small, f

    def _avg_peak_match_for_scale(
        self,
        frame_gray: np.ndarray,
        candidate_scale: float,
        preview_factor: float,
        sample_templates: List[np.ndarray],
    ) -> float:
        """Середнє з максимумів matchTemplate по підвибірці шаблонів (для ранжування scale)."""
        eff_scale = (
            candidate_scale * preview_factor
            if preview_factor < 0.999
            else candidate_scale
        )
        score_sum = 0.0
        used = 0
        for template in sample_templates:
            scaled = self._scale_template(template, eff_scale)
            h, w = scaled.shape[:2]
            if (
                h < 5
                or w < 5
                or h > frame_gray.shape[0]
                or w > frame_gray.shape[1]
            ):
                continue
            result = cv2.matchTemplate(frame_gray, scaled, cv2.TM_CCOEFF_NORMED)
            _min_v, max_v, _ml, _mm = cv2.minMaxLoc(result)
            score_sum += float(max_v)
            used += 1
        if used == 0:
            return -1.0
        return score_sum / float(used)

    def _best_scale_from_candidates(
        self,
        frame_gray: np.ndarray,
        preview_factor: float,
        candidates: List[float],
        sample_templates: List[np.ndarray],
    ) -> Tuple[float, float]:
        best_scale = self.template_scale
        best_avg = -1.0
        for candidate_scale in candidates:
            avg = self._avg_peak_match_for_scale(
                frame_gray,
                candidate_scale,
                preview_factor,
                sample_templates,
            )
            if avg > best_avg:
                best_avg = avg
                best_scale = candidate_scale
        return float(best_scale), float(best_avg)

    @staticmethod
    def _frange(a: float, b: float, step: float) -> List[float]:
        out: List[float] = []
        x = a
        while x <= b + 1e-9:
            out.append(round(float(x), 4))
            x += step
        return out

    def _estimate_best_scale(self, frame_gray: np.ndarray) -> float:
        """
        Швидкий підбір глобального scale: прев’ю кадр, мало шаблонів, грубо+тонко,
        або вузьке вікно навколо попереднього scale (авто-цикл).
        """
        min_scale, max_scale = self.auto_scale_range
        if min_scale <= 0 or max_scale <= 0 or max_scale < min_scale:
            return self.template_scale

        sample_templates = list(self.templates.values())[: self.auto_scale_sample_templates]
        if not sample_templates:
            return self.template_scale

        preview, preview_factor = self._gray_preview_for_scale_search(frame_gray)

        # 1) Повторне використання попереднього масштабу (типово наступний кадр тієї ж гри).
        if self.auto_scale_reuse and self._last_auto_scale is not None:
            c = float(self._last_auto_scale)
            if min_scale <= c <= max_scale:
                span = self.auto_scale_reuse_half_span
                reuse_candidates = self._frange(
                    max(min_scale, c - span),
                    min(max_scale, c + span),
                    self.auto_scale_fine_step,
                )
                if reuse_candidates:
                    rs, ra = self._best_scale_from_candidates(
                        preview,
                        preview_factor,
                        reuse_candidates,
                        sample_templates,
                    )
                    if ra >= self.auto_scale_reuse_min_avg:
                        return rs

        # 2) Грубий перебір по всьому діапазону.
        coarse_step = max(0.05, self.auto_scale_coarse_step)
        coarse_list = self._frange(min_scale, max_scale, coarse_step)
        best_c, _ = self._best_scale_from_candidates(
            preview, preview_factor, coarse_list, sample_templates
        )

        # 3) Точніше навколо найкращого грубого.
        fine_lo = max(min_scale, best_c - self.auto_scale_fine_half_span)
        fine_hi = min(max_scale, best_c + self.auto_scale_fine_half_span)
        fine_step = max(0.015, self.auto_scale_fine_step)
        fine_list = self._frange(fine_lo, fine_hi, fine_step)
        best_f, _ = self._best_scale_from_candidates(
            preview, preview_factor, fine_list, sample_templates
        )
        return float(best_f)

    @staticmethod
    def _scale_template(template: np.ndarray, scale: float) -> np.ndarray:
        if abs(scale - 1.0) < 1e-6:
            return template
        nh = max(1, int(round(template.shape[0] * scale)))
        nw = max(1, int(round(template.shape[1] * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(template, (nw, nh), interpolation=interp)

    @staticmethod
    def _gray_roi_inset(
        gray: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        inset_ratio: float,
    ) -> np.ndarray:
        """Виріз у відтінках сірого з відступом від рамки (менше рамки детектора)."""
        if inset_ratio > 0:
            dx = max(1, int(w * inset_ratio))
            dy = max(1, int(h * inset_ratio))
            x, y, w, h = x + dx, y + dy, w - 2 * dx, h - 2 * dy
        H, Wg = gray.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(Wg, x + max(1, w))
        y2 = min(H, y + max(1, h))
        if x2 <= x1 or y2 <= y1:
            return np.array([], dtype=np.uint8)
        return gray[y1:y2, x1:x2]

    def pair_patches_look_same(
        self,
        frame_bgr: np.ndarray,
        ax: int,
        ay: int,
        aw: int,
        ah: int,
        bx: int,
        by: int,
        bw: int,
        bh: int,
        inset_ratio: float = 0.08,
        compare_size: int = 80,
        min_normalized_score: float = 0.72,
    ) -> bool:
        """
        Чи виглядають дві області кадру як одна й та сама плитка (незалежно від імені шаблону).

        Потрібно, бо інколи WTA/шаблони дають однаковий tile_type різним малюнкам.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        a = self._gray_roi_inset(gray, ax, ay, aw, ah, inset_ratio)
        b = self._gray_roi_inset(gray, bx, by, bw, bh, inset_ratio)
        if a.size < 24 or b.size < 24:
            return False
        a_r = cv2.resize(
            a, (compare_size, compare_size), interpolation=cv2.INTER_AREA
        )
        b_r = cv2.resize(
            b, (compare_size, compare_size), interpolation=cv2.INTER_AREA
        )
        s1 = float(cv2.matchTemplate(a_r, b_r, cv2.TM_CCOEFF_NORMED)[0, 0])
        s2 = float(cv2.matchTemplate(b_r, a_r, cv2.TM_CCOEFF_NORMED)[0, 0])
        return max(s1, s2) >= min_normalized_score

    def analyze_once(
        self, region: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, List[TileMatch], Dict[str, List[TileMatch]], Tuple[int, int, int, int]]:
        """Один цикл: захват екрана + пошук шаблонів + прямокутник захвату для оверлею."""
        frame, cap_rect = self.capture_screen(region=region)
        matches, grouped = self.find_templates(frame)
        return frame, matches, grouped, cap_rect

    @staticmethod
    def draw_matches(frame_bgr: np.ndarray, matches: List[TileMatch]) -> np.ndarray:
        """Малює рамки навколо знайдених плиток (для дебагу)."""
        canvas = frame_bgr.copy()
        for match in matches:
            pt1 = (match.x, match.y)
            pt2 = (match.x + match.w, match.y + match.h)
            cv2.rectangle(canvas, pt1, pt2, (0, 255, 0), 2)
            label = f"{match.tile_type} {match.confidence:.2f}"
            cv2.putText(
                canvas,
                label,
                (match.x, max(0, match.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas

    @staticmethod
    def _non_max_suppression(
        matches: List[TileMatch], iou_threshold: float = 0.3
    ) -> List[TileMatch]:
        """Проста NMS, щоб прибрати дублікати збігів."""
        if not matches:
            return []

        boxes = np.array(
            [
                [m.x, m.y, m.x + m.w, m.y + m.h]
                for m in sorted(matches, key=lambda item: item.confidence, reverse=True)
            ],
            dtype=np.float32,
        )
        scores = np.array(
            [m.confidence for m in sorted(matches, key=lambda item: item.confidence, reverse=True)],
            dtype=np.float32,
        )
        ordered = sorted(matches, key=lambda item: item.confidence, reverse=True)

        keep_indices: List[int] = []
        idxs = np.arange(len(boxes))

        while len(idxs) > 0:
            current = idxs[0]
            keep_indices.append(int(current))
            if len(idxs) == 1:
                break

            rest = idxs[1:]
            xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter = inter_w * inter_h

            area_current = (boxes[current, 2] - boxes[current, 0]) * (
                boxes[current, 3] - boxes[current, 1]
            )
            area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (
                boxes[rest, 3] - boxes[rest, 1]
            )
            union = area_current + area_rest - inter + 1e-6
            iou = inter / union

            idxs = rest[iou <= iou_threshold]

        return [ordered[i] for i in keep_indices if scores[i] >= 0.0]

    @staticmethod
    def _winner_takes_all(
        matches: List[TileMatch], center_radius_ratio: float = 0.35
    ) -> List[TileMatch]:
        """
        Залишає одну найкращу мітку для кожної фізичної плитки.

        Ідея:
        - сортуємо всі збіги за score (від більшого до меншого);
        - якщо новий збіг має центр дуже близько до вже прийнятого,
          вважаємо, що це та сама плитка, і відкидаємо слабший.
        """
        if not matches:
            return []

        areas = [max(1.0, float(m.w * m.h)) for m in matches]
        median_area = float(np.median(np.array(areas, dtype=np.float32)))

        # Сортуємо за комбінованим quality: confidence * area_stability.
        # Це допомагає, коли в одній точці є кілька кандидатів з різним розміром рамки.
        ordered = sorted(
            matches,
            key=lambda item: VisionEngine._quality_score(item, median_area),
            reverse=True,
        )
        chosen: List[TileMatch] = []

        for candidate in ordered:
            cx = candidate.x + candidate.w / 2.0
            cy = candidate.y + candidate.h / 2.0
            radius = max(6.0, min(candidate.w, candidate.h) * center_radius_ratio)

            duplicate = False
            for kept in chosen:
                kx = kept.x + kept.w / 2.0
                ky = kept.y + kept.h / 2.0
                if abs(cx - kx) <= radius and abs(cy - ky) <= radius:
                    duplicate = True
                    break

            if not duplicate:
                chosen.append(candidate)

        return chosen

    @staticmethod
    def _quality_score(match: TileMatch, median_area: float) -> float:
        """
        Повертає quality для WTA:
        confidence * area_stability, де area_stability вищий для рамок,
        площа яких ближча до медіанної площі знайдених плиток.
        """
        area = max(1.0, float(match.w * match.h))
        ratio = area / max(1.0, median_area)
        # Симетричний штраф для занадто малих/великих рамок.
        deviation = abs(math.log(ratio))
        area_stability = 1.0 / (1.0 + 0.9 * deviation)
        return float(match.confidence) * area_stability

    @staticmethod
    def _group_by_type(matches: List[TileMatch]) -> Dict[str, List[TileMatch]]:
        """Групує фінальні збіги за типом плитки для зручного використання в UI/engine."""
        grouped: Dict[str, List[TileMatch]] = {}
        for match in matches:
            grouped.setdefault(match.tile_type, []).append(match)
        return grouped
