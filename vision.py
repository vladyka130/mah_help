from __future__ import annotations

"""
Розпізнавання плиток: OpenCV — основний інструмент.

Для пошуку однакових (типових) плиток на скріншоті використовується
``cv2.matchTemplate()`` (``TM_CCOEFF_NORMED``) по сірих шаблонах з ``assets/tiles``;
масштаб підлаштовується, далі NMS / WTA. Порівняння двох ROI пари — теж через
``matchTemplate`` (див. ``pair_patches_look_same``).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
import os
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
    # Середня яскравість ROI (0–255) після фільтра затемнення; 0 якщо не обчислювали.
    luma_mean: float = 0.0


class VisionEngine:
    """
    Захоплення екрана та пошук плиток.

    Ядро розпізнавання — OpenCV ``cv2.matchTemplate()``: для кожного типу плитки
    шаблон порівнюється з кадром у градаціях сірого; збіги вище порога дають
    кандидатів на ідентичні плитки на полі.

    Дефолти конструктора — підібрані «з коробки» (без панелі параметрів): баланс між
    хибними рамками на фоні, відсіканням затемнених фішок і знаходженням однакових пар.
    """

    def __init__(
        self,
        templates_dir: str = "assets/tiles",
        threshold: float = 0.55,
        template_scale: float = 1.0,
        auto_scale: bool = True,
        auto_scale_range: Tuple[float, float] = (0.52, 1.50),
        auto_scale_step: float = 0.05,
        # Швидкий підбір масштабу (грубий крок + уточнення; прев’ю кадр; повторне використання попереднього s).
        auto_scale_coarse_step: float = 0.12,
        auto_scale_fine_step: float = 0.035,
        auto_scale_fine_half_span: float = 0.12,
        auto_scale_sample_templates: int = 3,
        auto_scale_preview_max_side: int = 900,
        auto_scale_reuse: bool = True,
        auto_scale_reuse_half_span: float = 0.10,
        auto_scale_reuse_min_avg: float = 0.36,
        # Затемнені фішки (як у референсних скрінах assets/reference_boards): не прибираються грою — відсікаємо за яскравістю.
        filter_darkened_tiles: bool = True,
        luma_inset_ratio: float = 0.10,
        luma_min_cluster_separation: float = 7.5,
        luma_min_matches: int = 4,
        # Відкат luma-фільтра вимкнено за замовчуванням: інакше знову з’являються затемнені фішки.
        luma_rollback_min_keep_ratio: float = 0.0,
        # Після кластерів: відкинути детекції, де яскравість < (макс. на кадрі − це) — типово затемнені.
        luma_max_drop_from_brightest: float = 42.0,
        # Відсікання хибних збігів на рівному фоні (matchTemplate «бачить» текстуру там, де фішки немає).
        filter_low_texture_detections: bool = True,
        detection_texture_inset_ratio: float = 0.12,
        detection_min_gray_std: float = 11.5,
        detection_min_laplacian_var: float = 6.0,
        # Якщо після фільтра лишилось занадто мало збігів — скасувати його на цьому кадрі.
        detection_texture_rollback_min_keep_ratio: float = 0.42,
        # Пороги matchTemplate між вирізами двох плиток (різні для «той самий шаблон» / різні типи / висока впевненість шаблону).
        # Параметри сірого ROI + pair_edge_min_score (градієнти) разом зменшують плутанину «однаковий фон».
        # Баланс: занадто високо — не знаходить ту саму пару при різному світлі на полі.
        pair_visual_same_type: float = 0.48,
        pair_visual_cross_type: float = 0.41,
        # Augment: різні імена файлів шаблонів, один малюнок (наприклад дві Анубіса з різних PNG).
        pair_visual_cross_augment: float = 0.37,
        pair_trust_template_conf: float = 0.82,
        pair_visual_if_trusted: float = 0.42,
        # Майже вимкнено: ім’я шаблону/число в matchTemplate не замінює порівняння візерунка.
        pair_skip_visual_min_conf: float = 0.995,
        # Пропуск ROI лише при дуже високому збігу шаблону (обидва ≥ порога). 1.05 = ніколи не пропускати.
        pair_skip_roi_same_type_min_conf: float = 0.94,
        # Кадр візерунка: відрізати рамку 3D; augment теж зосереджений на центрі, не на однаковому «борту».
        pair_augment_inset_ratio: float = 0.10,
        # Порівняння пар: глибший inset — менше ваги країв / однакових кутів (напр. жуки по кутах vs центр).
        pair_default_inset_ratio: float = 0.13,
        pair_compare_use_clahe: bool = True,
        pair_compare_size: int = 56,
        # Друга перевірка пари: збіг градієнтів (контури символу). Різні іконки на однаковому тлі дають низький edge-score.
        # Градієнти: різні символи на схожому тлі рідше проходять обидва пороги.
        pair_edge_gate_enabled: bool = True,
        pair_edge_min_score: float = 0.33,
        # Додатковий виріз «лише центр символу»: відсікає пари з однаковими краями й різним центром (5-й жук, інший ранг бамбука).
        pair_center_gate_enabled: bool = True,
        pair_center_extra_inset: float = 0.10,
        pair_center_min_score_slack: float = 0.055,
        # Верхній лівий кут (цифра рангу бамбука тощо): якщо шаблон помилково дає один tile_type на різні фішки.
        pair_corner_gate_enabled: bool = True,
        pair_corner_width_ratio: float = 0.36,
        pair_corner_height_ratio: float = 0.32,
        pair_corner_min_norm_score: float = 0.70,
        # Кольоровий центр (HSV H–S гістограма): різні сітки «гачків» / червоний vs синій центр — не одна фішка.
        pair_hsv_hist_gate_enabled: bool = True,
        pair_hsv_hist_center_inset: float = 0.12,
        pair_hsv_hist_min_correl: float = 0.64,
        # Злиття детекцій: строгий прохід + м’який (більше фішок на полі без очікування «нуля пар»).
        merge_relaxed_delta: float = 0.11,
        merge_relaxed_floor: float = 0.37,
        # Паралельний matchTemplate по різних шаблонах (якщо шаблонів достатньо).
        parallel_template_workers: int = 0,
        min_templates_for_parallel: int = 6,
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
        self.filter_darkened_tiles = bool(filter_darkened_tiles)
        self.luma_inset_ratio = float(luma_inset_ratio)
        self.luma_min_cluster_separation = float(luma_min_cluster_separation)
        self.luma_min_matches = max(4, int(luma_min_matches))
        self.luma_rollback_min_keep_ratio = float(luma_rollback_min_keep_ratio)
        self.luma_max_drop_from_brightest = float(luma_max_drop_from_brightest)
        self.filter_low_texture_detections = bool(filter_low_texture_detections)
        self.detection_texture_inset_ratio = float(detection_texture_inset_ratio)
        self.detection_min_gray_std = float(detection_min_gray_std)
        self.detection_min_laplacian_var = float(detection_min_laplacian_var)
        self.detection_texture_rollback_min_keep_ratio = float(
            detection_texture_rollback_min_keep_ratio
        )
        self.pair_visual_same_type = float(pair_visual_same_type)
        self.pair_visual_cross_type = float(pair_visual_cross_type)
        self.pair_trust_template_conf = float(pair_trust_template_conf)
        self.pair_visual_if_trusted = float(pair_visual_if_trusted)
        self.pair_visual_cross_augment = float(pair_visual_cross_augment)
        self.pair_skip_visual_min_conf = float(pair_skip_visual_min_conf)
        self.pair_skip_roi_same_type_min_conf = float(pair_skip_roi_same_type_min_conf)
        self.pair_augment_inset_ratio = float(pair_augment_inset_ratio)
        self.pair_default_inset_ratio = float(pair_default_inset_ratio)
        self.pair_compare_use_clahe = bool(pair_compare_use_clahe)
        self.pair_compare_size = max(32, int(pair_compare_size))
        self.pair_edge_gate_enabled = bool(pair_edge_gate_enabled)
        self.pair_edge_min_score = float(pair_edge_min_score)
        self.pair_center_gate_enabled = bool(pair_center_gate_enabled)
        self.pair_center_extra_inset = float(pair_center_extra_inset)
        self.pair_center_min_score_slack = float(pair_center_min_score_slack)
        self.pair_corner_gate_enabled = bool(pair_corner_gate_enabled)
        self.pair_corner_width_ratio = float(pair_corner_width_ratio)
        self.pair_corner_height_ratio = float(pair_corner_height_ratio)
        self.pair_corner_min_norm_score = float(pair_corner_min_norm_score)
        self.pair_hsv_hist_gate_enabled = bool(pair_hsv_hist_gate_enabled)
        self.pair_hsv_hist_center_inset = float(pair_hsv_hist_center_inset)
        self.pair_hsv_hist_min_correl = float(pair_hsv_hist_min_correl)
        self.merge_relaxed_delta = float(merge_relaxed_delta)
        self.merge_relaxed_floor = float(merge_relaxed_floor)
        pw = int(parallel_template_workers)
        self.parallel_template_workers = pw if pw > 0 else max(
            1, min(8, (os.cpu_count() or 4))
        )
        self.min_templates_for_parallel = max(3, int(min_templates_for_parallel))
        # М’якший CLAHE — менше штучних відмінностей між двома копіями однієї фішки.
        self._clahe_pair = cv2.createCLAHE(clipLimit=1.65, tileGridSize=(8, 8))
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

    def _detect_single_template(
        self,
        tile_type: str,
        template: np.ndarray,
        frame_gray: np.ndarray,
        scale: float,
        eff_threshold: float,
    ) -> List[TileMatch]:
        """Один шаблон на кадрі + локальний NMS (для паралельного запуску)."""
        scaled_template = self._scale_template(template, scale)
        h, w = scaled_template.shape[:2]
        if h < 5 or w < 5 or h > frame_gray.shape[0] or w > frame_gray.shape[1]:
            return []
        result = cv2.matchTemplate(frame_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= eff_threshold)
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
        return self._non_max_suppression(tile_matches, iou_threshold=0.30)

    def find_templates(
        self,
        frame_bgr: np.ndarray,
        threshold: Optional[float] = None,
        apply_darkened_filter: Optional[bool] = None,
        reuse_cached_scale: bool = False,
    ) -> Tuple[List[TileMatch], Dict[str, List[TileMatch]]]:
        """
        Шукає всі типи плиток на кадрі.

        Для кожного шаблону — ``cv2.matchTemplate(..., TM_CCOEFF_NORMED)`` по всьому кадру;
        це стандартний спосіб знайти ділянки, візуально збігаються з еталоном плитки.

        ``threshold`` — перевизначити поріг збігу; ``apply_darkened_filter`` — чи застосовувати відсікання затемнених.
        ``reuse_cached_scale`` — без повторного підбору масштабу (після першого успішного кадру в автогра).
        """
        eff_threshold = float(self.threshold if threshold is None else threshold)
        use_luma = (
            self.filter_darkened_tiles
            if apply_darkened_filter is None
            else bool(apply_darkened_filter)
        )
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scale = self.template_scale
        if self.auto_scale:
            if reuse_cached_scale and self._last_auto_scale is not None:
                scale = float(self._last_auto_scale)
            else:
                scale = self._estimate_best_scale(frame_gray)
                self._last_auto_scale = scale

        all_matches: List[TileMatch] = []
        template_items = list(self.templates.items())
        use_parallel = len(template_items) >= self.min_templates_for_parallel

        if use_parallel:
            workers = min(self.parallel_template_workers, len(template_items))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        self._detect_single_template,
                        tt,
                        tpl,
                        frame_gray,
                        scale,
                        eff_threshold,
                    )
                    for tt, tpl in template_items
                ]
                for fu in as_completed(futures):
                    all_matches.extend(fu.result())
        else:
            for tile_type, template in template_items:
                all_matches.extend(
                    self._detect_single_template(
                        tile_type,
                        template,
                        frame_gray,
                        scale,
                        eff_threshold,
                    )
                )

        # Глобальний NMS між усіма класами: прибирає рамки, що сильно накладаються.
        global_nms = self._non_max_suppression(all_matches, iou_threshold=0.35)
        # Winner-takes-all: для кожної фізичної плитки залишаємо лише мітку з найвищим score.
        final_matches = self._winner_takes_all(global_nms, center_radius_ratio=0.35)
        if self.filter_low_texture_detections:
            final_matches = self._filter_low_texture_matches(frame_gray, final_matches)
        if use_luma:
            final_matches = self._filter_reference_style_dimmed(
                frame_gray, final_matches
            )
        grouped = self._group_by_type(final_matches)
        return final_matches, grouped

    def find_templates_merged(
        self,
        frame_bgr: np.ndarray,
        threshold: Optional[float] = None,
        relaxed_delta: Optional[float] = None,
        relaxed_floor: Optional[float] = None,
    ) -> Tuple[List[TileMatch], Dict[str, List[TileMatch]]]:
        """
        Два проходи шаблону: основний (із затемненням як у ``find_templates``) +
        м’який без фільтра яскравості. Об’єднує списки, потім NMS + WTA.

        Так знаходяться фішки на краях поля / при іншому світлі, навіть коли вже
        є інші детекції (раніше м’який поріг викликався лише якщо пар не було взагалі).
        """
        eff = float(self.threshold if threshold is None else threshold)
        rd = (
            self.merge_relaxed_delta
            if relaxed_delta is None
            else float(relaxed_delta)
        )
        rf = (
            self.merge_relaxed_floor if relaxed_floor is None else float(relaxed_floor)
        )
        m_strict, _ = self.find_templates(
            frame_bgr,
            threshold=eff,
            apply_darkened_filter=None,
        )
        th_lo = max(rf, eff - rd)
        m_loose, _ = self.find_templates(
            frame_bgr,
            threshold=th_lo,
            apply_darkened_filter=False,
        )
        combined = m_strict + m_loose
        global_nms = self._non_max_suppression(combined, iou_threshold=0.35)
        final_matches = self._winner_takes_all(global_nms, center_radius_ratio=0.35)
        grouped = self._group_by_type(final_matches)
        if final_matches:
            return final_matches, grouped
        # Обидва шари порожні (поріг / масштаб / світло) — один запасний прохід без затемнення.
        th_rescue = max(0.30, min(0.42, float(eff) - 0.22))
        return self.find_templates(
            frame_bgr,
            threshold=float(th_rescue),
            apply_darkened_filter=False,
        )

    def _tile_match_mean_luma(
        self, frame_gray: np.ndarray, match: TileMatch
    ) -> float:
        """Середня яскравість сірої внутрішньої частини рамки (без країв шаблону)."""
        inset = self.luma_inset_ratio
        x, y, w, h = match.x, match.y, match.w, match.h
        dx = max(1, int(round(w * inset)))
        dy = max(1, int(round(h * inset)))
        x1, y1 = x + dx, y + dy
        x2, y2 = x + w - dx, y + h - dy
        hg, wg = frame_gray.shape[:2]
        x1 = max(0, min(x1, wg - 1))
        y1 = max(0, min(y1, hg - 1))
        x2 = max(x1 + 1, min(x2, wg))
        y2 = max(y1 + 1, min(y2, hg))
        patch = frame_gray[y1:y2, x1:x2]
        if patch.size == 0:
            y2o = min(y + h, hg)
            x2o = min(x + w, wg)
            patch = frame_gray[y:y2o, x:x2o]
        if patch.size == 0:
            return 0.0
        return float(np.mean(patch))

    def _luma_keep_bright_cluster_mask(
        self, lumas: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Двокластерний поділ за яскравістю (k-means у 1D).
        Повертає маску True для плиток «яскравішого» кластера або None, якщо розділення ненадійне.
        """
        v = lumas.astype(np.float64)
        n = len(v)
        if n < self.luma_min_matches:
            return None
        lo = float(np.percentile(v, 22))
        hi = float(np.percentile(v, 78))
        if hi - lo < 1.5:
            lo, hi = float(v.min()), float(v.max())
        min_sep = self.luma_min_cluster_separation

        for _ in range(60):
            dist_lo = np.abs(v - lo)
            dist_hi = np.abs(v - hi)
            mask_lo = dist_lo < dist_hi
            same = dist_lo == dist_hi
            if np.any(same):
                mask_lo = np.where(same, True, mask_lo)
            if np.sum(mask_lo) == 0 or np.sum(~mask_lo) == 0:
                return None
            lo_new = float(v[mask_lo].mean())
            hi_new = float(v[~mask_lo].mean())
            if hi_new < lo_new:
                lo_new, hi_new = hi_new, lo_new
            shift = abs(lo_new - lo) + abs(hi_new - hi)
            lo, hi = lo_new, hi_new
            if shift < 1e-4:
                break

        bright_c = max(lo, hi)
        dark_c = min(lo, hi)
        if bright_c - dark_c < min_sep:
            return None

        d_lo = np.abs(v - dark_c)
        d_hi = np.abs(v - bright_c)
        # Залишаємо кластер ближчий до яскравішого центроїду (затемнені — до нижнього).
        return d_hi < d_lo

    def _luma_keep_bright_by_gap(self, lumas: np.ndarray) -> Optional[np.ndarray]:
        """
        Резерв, якщо k-means не розділив: шукаємо найбільший розрив у відсортованих яскравостях
        (типово між «темними» недоступними й світлими доступними фішками).
        """
        if len(lumas) < 4:
            return None
        v = np.sort(lumas.astype(np.float64))
        gaps = np.diff(v)
        gi = int(np.argmax(gaps))
        if float(gaps[gi]) < 10.5:
            return None
        thresh = float((v[gi] + v[gi + 1]) / 2.0)
        return lumas >= thresh

    def _luma_keep_bright_fallback_percentile(self, lumas: np.ndarray) -> np.ndarray:
        """Останній резерв: прибирає найнижчі ~45% за середньою яскравістю ROI."""
        cut = float(np.percentile(lumas.astype(np.float64), 45))
        return lumas >= cut

    def _filter_reference_style_dimmed(
        self,
        frame_gray: np.ndarray,
        matches: List[TileMatch],
    ) -> List[TileMatch]:
        """
        Прибирає зі списку фішки з сильним затемненням (гейм-дизайн: недоступні хід).

        Орієнтир — референсні скріни в assets/reference_boards: там видно різницю яскравості
        між доступними та закритими плитками.
        """
        if not matches:
            return matches
        lumas_list = [self._tile_match_mean_luma(frame_gray, m) for m in matches]
        lumas = np.array(lumas_list, dtype=np.float64)
        keep_mask = self._luma_keep_bright_cluster_mask(lumas)
        if keep_mask is None:
            keep_mask = self._luma_keep_bright_by_gap(lumas)
        if keep_mask is None:
            keep_mask = self._luma_keep_bright_fallback_percentile(lumas)
        out: List[TileMatch] = []
        for i, m in enumerate(matches):
            if not keep_mask[i]:
                continue
            out.append(
                TileMatch(
                    tile_type=m.tile_type,
                    confidence=m.confidence,
                    x=m.x,
                    y=m.y,
                    w=m.w,
                    h=m.h,
                    luma_mean=float(lumas_list[i]),
                )
            )
        # Відсікаємо занадто темні порівняно з найяскравішою детекцією на кадрі.
        drop = float(self.luma_max_drop_from_brightest)
        if out and drop > 1.0:
            peak = float(np.max(lumas))
            floor_l = peak - drop
            out = [m for m in out if m.luma_mean >= floor_l]
        # Відкат лише якщо явно увімкнено (luma_rollback_min_keep_ratio > 0), щоб не повертати затемнені.
        ratio = len(out) / float(len(matches)) if matches else 1.0
        rb = float(self.luma_rollback_min_keep_ratio)
        if (
            rb > 1e-6
            and out
            and len(matches) >= 10
            and ratio < rb
        ):
            return [
                TileMatch(
                    tile_type=m.tile_type,
                    confidence=m.confidence,
                    x=m.x,
                    y=m.y,
                    w=m.w,
                    h=m.h,
                    luma_mean=lumas_list[i],
                )
                for i, m in enumerate(matches)
            ]
        return out if out else matches

    def _filter_low_texture_matches(
        self,
        frame_gray: np.ndarray,
        matches: List[TileMatch],
    ) -> List[TileMatch]:
        """
        Відсікає хибні збіги matchTemplate на рівному фоні (однорідна підлога / декор без фішки).

        Критерій: всередині ROI (без товстої рамки) має бути достатньо контрасту (std сірого)
        або варіації Лапласіана (дрібна текстура символу).
        Якщо після відсікання лишається занадто мало точок — скасовуємо фільтр на цьому кадрі.
        """
        if not matches:
            return matches
        inset = max(0.0, min(0.35, float(self.detection_texture_inset_ratio)))
        min_std = float(self.detection_min_gray_std)
        min_lap = float(self.detection_min_laplacian_var)
        hg, wg = frame_gray.shape[0], frame_gray.shape[1]
        out: List[TileMatch] = []
        for m in matches:
            x, y, w, h = int(m.x), int(m.y), int(m.w), int(m.h)
            dx = max(1, int(w * inset))
            dy = max(1, int(h * inset))
            x1, y1 = x + dx, y + dy
            x2, y2 = x + w - dx, y + h - dy
            x1 = max(0, min(x1, wg - 1))
            y1 = max(0, min(y1, hg - 1))
            x2 = max(x1 + 1, min(x2, wg))
            y2 = max(y1 + 1, min(y2, hg))
            roi = frame_gray[y1:y2, x1:x2]
            if roi.size < 24:
                continue
            sd = float(np.std(roi))
            lap = cv2.Laplacian(roi, cv2.CV_64F)
            lv = float(lap.var())
            if sd >= min_std or lv >= min_lap:
                out.append(m)
        ratio = len(out) / float(len(matches)) if matches else 1.0
        rb = float(self.detection_texture_rollback_min_keep_ratio)
        if out and len(matches) >= 8 and ratio < rb:
            return matches
        return out if out else matches

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

    @staticmethod
    def _bgr_roi_inset(
        bgr: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        inset_ratio: float,
    ) -> np.ndarray:
        """Виріз BGR з inset від рамки детектора (для кольорової перевірки центру)."""
        if inset_ratio > 0:
            dx = max(1, int(w * inset_ratio))
            dy = max(1, int(h * inset_ratio))
            x, y, w, h = x + dx, y + dy, w - 2 * dx, h - 2 * dy
        H, Wg = bgr.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(Wg, x + max(1, w))
        y2 = min(H, y + max(1, h))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return bgr[y1:y2, x1:x2]

    def _pair_hsv_center_hist_correl(
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
        inset_ratio: float,
    ) -> float:
        """
        Кореляція 2D гістограм відтінок–насиченість у центрі двох ROI.

        Повертає ``1.0``, якщо ROI замалі (перевірку пропускаємо). Інакше OpenCV HISTCMP_CORREL.
        """
        ra = self._bgr_roi_inset(frame_bgr, ax, ay, aw, ah, inset_ratio)
        rb = self._bgr_roi_inset(frame_bgr, bx, by, bw, bh, inset_ratio)
        if ra.size < 120 or rb.size < 120:
            return 1.0
        hsv_a = cv2.cvtColor(ra, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(rb, cv2.COLOR_BGR2HSV)
        h_bins, s_bins = 28, 18
        rng = [0, 180, 0, 256]
        ha = cv2.calcHist([hsv_a], [0, 1], None, [h_bins, s_bins], rng)
        hb = cv2.calcHist([hsv_b], [0, 1], None, [h_bins, s_bins], rng)
        cv2.normalize(ha, ha, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hb, hb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cr = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
        if not math.isfinite(cr):
            return 0.0
        return cr

    def _pair_edge_match_score(
        self, a_u8: np.ndarray, b_u8: np.ndarray, cs: int
    ) -> float:
        """Збіг карт градієнтів (контури символу), стійкіший до спільного кольору тла."""
        if a_u8.size < 9 or b_u8.size < 9:
            return 0.0
        gx = cv2.Sobel(a_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(a_u8, cv2.CV_32F, 0, 1, ksize=3)
        ea = cv2.magnitude(gx, gy)
        gx2 = cv2.Sobel(b_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy2 = cv2.Sobel(b_u8, cv2.CV_32F, 0, 1, ksize=3)
        eb = cv2.magnitude(gx2, gy2)
        ea_u8 = cv2.normalize(ea, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        eb_u8 = cv2.normalize(eb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ar = cv2.resize(ea_u8, (cs, cs), interpolation=cv2.INTER_AREA)
        br = cv2.resize(eb_u8, (cs, cs), interpolation=cv2.INTER_AREA)
        s1 = float(cv2.matchTemplate(ar, br, cv2.TM_CCOEFF_NORMED)[0, 0])
        s2 = float(cv2.matchTemplate(br, ar, cv2.TM_CCOEFF_NORMED)[0, 0])
        return max(s1, s2)

    def pair_top_left_corners_look_same(
        self,
        frame_gray: np.ndarray,
        ax: int,
        ay: int,
        aw: int,
        ah: int,
        bx: int,
        by: int,
        bw: int,
        bh: int,
    ) -> bool:
        """
        Швидке порівняння верхнього лівого фрагмента двох фішок (маленька цифра / маркер рангу).

        Якщо різні ранги бамбука (2 vs 5) помилково отримали один і той самий tile_type,
        кути не збігаються — пару не можна приймати лише за високим score шаблону.
        """
        if not self.pair_corner_gate_enabled:
            return True
        Hg, Wg = int(frame_gray.shape[0]), int(frame_gray.shape[1])
        rwa = max(6, int(aw * float(self.pair_corner_width_ratio)))
        rha = max(6, int(ah * float(self.pair_corner_height_ratio)))
        rwb = max(6, int(bw * float(self.pair_corner_width_ratio)))
        rhb = max(6, int(bh * float(self.pair_corner_height_ratio)))
        x1a, y1a = max(0, ax), max(0, ay)
        x2a, y2a = min(Wg, ax + rwa), min(Hg, ay + rha)
        x1b, y1b = max(0, bx), max(0, by)
        x2b, y2b = min(Wg, bx + rwb), min(Hg, by + rhb)
        if x2a <= x1a or y2a <= y1a or x2b <= x1b or y2b <= y1b:
            return True
        pa = frame_gray[y1a:y2a, x1a:x2a]
        pb = frame_gray[y1b:y2b, x1b:x2b]
        if pa.size < 36 or pb.size < 36:
            return True
        cs = 28
        a_r = cv2.resize(pa, (cs, cs), interpolation=cv2.INTER_AREA)
        b_r = cv2.resize(pb, (cs, cs), interpolation=cv2.INTER_AREA)
        s1 = float(cv2.matchTemplate(a_r, b_r, cv2.TM_CCOEFF_NORMED)[0, 0])
        s2 = float(cv2.matchTemplate(b_r, a_r, cv2.TM_CCOEFF_NORMED)[0, 0])
        return max(s1, s2) >= float(self.pair_corner_min_norm_score)

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
        compare_size: Optional[int] = None,
        min_normalized_score: float = 0.72,
        allow_narrow_retry: bool = True,
        edge_min_score: Optional[float] = None,
        frame_gray: Optional[np.ndarray] = None,
        fast: bool = False,
    ) -> bool:
        """
        Чи виглядають дві області кадру як одна й та сама плитка (незалежно від імені шаблону).

        Потрібно, бо інколи WTA/шаблони дають однаковий tile_type різним малюнкам.
        Порівняння — ``matchTemplate`` на ROI + окремо на карті градієнтів (менше хибних
        пар через однаковий колір тла без однакового символу).

        Якщо перший виріз не пройшов — повтор із вужчим inset; на retry поріг сірого трохи
        послаблюється; поріг градієнтів теж трохи знижується на retry.
        Якщо CLAHE розводить два центри однієї фішки — додаткова спроба на сірому без CLAHE.

        ``fast=True`` — один прохід CLAHE + без вузького retry (лише для прискореного автогра).
        """
        if fast:
            allow_narrow_retry = False
        emin = (
            float(edge_min_score)
            if edge_min_score is not None
            else float(self.pair_edge_min_score)
        )
        # Один раз на кадр передавати frame_gray з main — інакше сотні повних cvtColor на кожному аналізі.
        gray = (
            frame_gray
            if frame_gray is not None
            else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        )
        a = self._gray_roi_inset(gray, ax, ay, aw, ah, inset_ratio)
        b = self._gray_roi_inset(gray, bx, by, bw, bh, inset_ratio)
        if a.size < 24 or b.size < 24:
            if allow_narrow_retry and inset_ratio > 0.028:
                narrow = max(0.02, inset_ratio * 0.42)
                thr_retry = max(0.42, float(min_normalized_score) - 0.02)
                em_retry = max(0.30, emin - 0.025)
                return self.pair_patches_look_same(
                    frame_bgr,
                    ax,
                    ay,
                    aw,
                    ah,
                    bx,
                    by,
                    bw,
                    bh,
                    inset_ratio=narrow,
                    compare_size=compare_size,
                    min_normalized_score=thr_retry,
                    allow_narrow_retry=False,
                    edge_min_score=em_retry,
                    frame_gray=gray,
                    fast=fast,
                )
            return False
        cs = self.pair_compare_size if compare_size is None else int(compare_size)
        if fast:
            cs = max(32, min(cs, 48))

        def _pair_match_score(a_u8: np.ndarray, b_u8: np.ndarray, use_clahe: bool) -> float:
            aa, bb = a_u8, b_u8
            if use_clahe and self.pair_compare_use_clahe:
                aa = self._clahe_pair.apply(aa)
                bb = self._clahe_pair.apply(bb)
            a_r = cv2.resize(aa, (cs, cs), interpolation=cv2.INTER_AREA)
            b_r = cv2.resize(bb, (cs, cs), interpolation=cv2.INTER_AREA)
            s1 = float(cv2.matchTemplate(a_r, b_r, cv2.TM_CCOEFF_NORMED)[0, 0])
            s2 = float(cv2.matchTemplate(b_r, a_r, cv2.TM_CCOEFF_NORMED)[0, 0])
            return max(s1, s2)

        best = _pair_match_score(a, b, use_clahe=True)
        if self.pair_compare_use_clahe and not fast:
            best = max(best, _pair_match_score(a, b, use_clahe=False))
        edge_best = self._pair_edge_match_score(a, b, cs)
        gray_ok = best >= min_normalized_score
        edge_ok = (not self.pair_edge_gate_enabled) or (edge_best >= emin)
        ok = gray_ok and edge_ok
        # Вузький центр: однакові кути поля / символу, але різний центр (напр. червоний жук посередині).
        # У fast-тurbo теж увімкнено (лише один прохід CLAHE через гілку нижче), щоб автогра не клікала «чужі» пари.
        if ok and self.pair_center_gate_enabled:
            ir_c = min(
                0.36,
                float(inset_ratio) + float(self.pair_center_extra_inset),
            )
            ac = self._gray_roi_inset(gray, ax, ay, aw, ah, ir_c)
            bc = self._gray_roi_inset(gray, bx, by, bw, bh, ir_c)
            if ac.size >= 24 and bc.size >= 24:
                c_thr = max(
                    0.40,
                    float(min_normalized_score)
                    - float(self.pair_center_min_score_slack),
                )
                c_best = _pair_match_score(ac, bc, use_clahe=True)
                if self.pair_compare_use_clahe and not fast:
                    c_best = max(c_best, _pair_match_score(ac, bc, use_clahe=False))
                c_edge = self._pair_edge_match_score(ac, bc, cs)
                c_gray_ok = c_best >= c_thr
                c_edge_ok = (not self.pair_edge_gate_enabled) or (c_edge >= emin)
                ok = c_gray_ok and c_edge_ok
        # Центр у кольорі HSV: сітка 9 vs 7 «гачків», червоний стовпчик vs синій — інакша гістограма H–S.
        if ok and self.pair_hsv_hist_gate_enabled:
            ir_h = min(
                0.42,
                float(inset_ratio) + float(self.pair_hsv_hist_center_inset),
            )
            cr = self._pair_hsv_center_hist_correl(
                frame_bgr, ax, ay, aw, ah, bx, by, bw, bh, ir_h
            )
            if cr < float(self.pair_hsv_hist_min_correl):
                ok = False
        if ok:
            return True
        if allow_narrow_retry and inset_ratio > 0.028:
            narrow = max(0.02, inset_ratio * 0.42)
            thr_retry = max(0.42, float(min_normalized_score) - 0.02)
            em_retry = max(0.30, emin - 0.025)
            return self.pair_patches_look_same(
                frame_bgr,
                ax,
                ay,
                aw,
                ah,
                bx,
                by,
                bw,
                bh,
                inset_ratio=narrow,
                compare_size=compare_size,
                min_normalized_score=thr_retry,
                allow_narrow_retry=False,
                edge_min_score=em_retry,
                frame_gray=gray,
                fast=fast,
            )
        return False

    def analyze_once(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        template_threshold: Optional[float] = None,
        apply_darkened_filter: Optional[bool] = None,
    ) -> Tuple[np.ndarray, List[TileMatch], Dict[str, List[TileMatch]], Tuple[int, int, int, int]]:
        """Один цикл: захват екрана + пошук шаблонів + прямокутник захвату для оверлею."""
        frame, cap_rect = self.capture_screen(region=region)
        matches, grouped = self.find_templates(
            frame,
            threshold=template_threshold,
            apply_darkened_filter=apply_darkened_filter,
        )
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
