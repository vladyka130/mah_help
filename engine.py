from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Dict, List, Optional, Tuple
from urllib import error, request

from vision import TileMatch


@dataclass
class EngineTile:
    """Плитка в контексті логіки гри."""

    id: int
    tile_type: str
    x: int
    y: int
    w: int
    h: int
    # Якість збігу шаблону з Vision (щоб не зводити пари зі слабкими хибними збігами).
    confidence: float = 0.0

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class PairCandidate:
    """Кандидат пари для видалення."""

    pair_id: str
    tile_type: str
    first_id: int
    second_id: int
    first_coords: Tuple[int, int]
    second_coords: Tuple[int, int]
    # Точні w/h з Vision/детекції (для оверлея без костилів).
    first_w: int
    first_h: int
    second_w: int
    second_h: int
    unlock_score: int


class MahjongEngine:
    """
    Логіка маджонгу:
    - побудова відносного положення плиток;
    - визначення вільних плиток;
    - формування пар вільних плиток.
    """

    def __init__(
        self,
        side_overlap_ratio: float = 0.26,
        top_overlap_ratio: float = 0.18,
        center_line_tolerance_ratio: float = 0.50,
        z_offset: Tuple[int, int] = (-6, -6),
        pair_min_confidence: float = 0.54,
        # Якщо одна верхня плитка лежить на кількох нижніх, кожна нижня має мало площі
        # під перекриттям — додатковий шлях «часткове накриття».
        top_overlap_partial_min: float = 0.042,
        top_min_cover_w_ratio: float = 0.048,
        top_min_cover_h_ratio: float = 0.038,
        # Горизонтальний зазор між «торцями» сусідів (частина ширини плитки), ізометрія.
        side_max_gap_ratio: float = 0.46,
        # Якщо строге перетинання по Y не досягнуте, але центри близько по вертикалі.
        side_overlap_loose_min: float = 0.11,
        side_max_center_dy_ratio: float = 0.58,
    ) -> None:
        self.side_overlap_ratio = side_overlap_ratio
        self.top_overlap_ratio = top_overlap_ratio
        self.center_line_tolerance_ratio = center_line_tolerance_ratio
        # Для 3D-подібних ігор верхня плитка часто зміщена відносно нижньої.
        self.z_offset = z_offset
        self.pair_min_confidence = float(pair_min_confidence)
        self.top_overlap_partial_min = float(top_overlap_partial_min)
        self.top_min_cover_w_ratio = float(top_min_cover_w_ratio)
        self.top_min_cover_h_ratio = float(top_min_cover_h_ratio)
        self.side_max_gap_ratio = float(side_max_gap_ratio)
        self.side_overlap_loose_min = float(side_overlap_loose_min)
        self.side_max_center_dy_ratio = float(side_max_center_dy_ratio)

    def build_tiles(self, matches: List[TileMatch]) -> List[EngineTile]:
        """Конвертує детекції Vision у структури Engine."""
        tiles: List[EngineTile] = []
        for idx, m in enumerate(matches):
            tiles.append(
                EngineTile(
                    id=idx,
                    tile_type=m.tile_type,
                    x=int(m.x),
                    y=int(m.y),
                    w=int(m.w),
                    h=int(m.h),
                    confidence=float(m.confidence),
                )
            )
        return tiles

    def build_relations(self, tiles: List[EngineTile]) -> Dict[int, Dict[str, List[int]]]:
        """
        Будує відносини між плитками:
        - left: сусіди зліва;
        - right: сусіди справа;
        - top: плитки, що перекривають зверху.
        """
        relations: Dict[int, Dict[str, List[int]]] = {
            t.id: {"left": [], "right": [], "top": []} for t in tiles
        }

        for tile in tiles:
            for other in tiles:
                if tile.id == other.id:
                    continue

                if self._is_left_neighbor(tile, other):
                    relations[tile.id]["left"].append(other.id)
                if self._is_right_neighbor(tile, other):
                    relations[tile.id]["right"].append(other.id)
                if self._is_top_blocker(tile, other):
                    relations[tile.id]["top"].append(other.id)

        return relations

    def find_free_tiles(
        self, tiles: List[EngineTile], relations: Dict[int, Dict[str, List[int]]]
    ) -> List[EngineTile]:
        """
        Плитка вільна, якщо:
        - НЕ перекрита зверху;
        - і має вільний хоча б один бік (немає лівого або немає правого сусіда).
        """
        free: List[EngineTile] = []
        for tile in tiles:
            rel = relations[tile.id]
            has_top = len(rel["top"]) > 0
            has_left = len(rel["left"]) > 0
            has_right = len(rel["right"]) > 0
            if (not has_top) and ((not has_left) or (not has_right)):
                free.append(tile)
        return free

    def find_free_pairs(
        self, tiles: List[EngineTile], relations: Dict[int, Dict[str, List[int]]]
    ) -> List[PairCandidate]:
        """Повертає всі пари однакових вільних плиток."""
        free_tiles = self.find_free_tiles(tiles, relations)
        by_type: Dict[str, List[EngineTile]] = {}
        for tile in free_tiles:
            by_type.setdefault(tile.tile_type, []).append(tile)

        pairs: List[PairCandidate] = []
        for tile_type, group in by_type.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a = group[i]
                    b = group[j]
                    if (
                        min(a.confidence, b.confidence)
                        < self.pair_min_confidence
                    ):
                        continue
                    pair_id = f"{tile_type}:{a.id}-{b.id}"
                    unlock_score = self._estimate_unlock_score((a, b), tiles, relations)
                    pairs.append(
                        PairCandidate(
                            pair_id=pair_id,
                            tile_type=tile_type,
                            first_id=a.id,
                            second_id=b.id,
                            first_coords=(a.x, a.y),
                            second_coords=(b.x, b.y),
                            first_w=int(a.w),
                            first_h=int(a.h),
                            second_w=int(b.w),
                            second_h=int(b.h),
                            unlock_score=unlock_score,
                        )
                    )

        # Кращі кандидати зверху: вищий unlock_score, потім ближче до центра.
        pairs.sort(key=lambda p: p.unlock_score, reverse=True)
        return pairs

    def augment_free_pairs_cross_type_visual(
        self,
        pairs: List[PairCandidate],
        tiles: List[EngineTile],
        relations: Dict[int, Dict[str, List[int]]],
        tiles_look_same: Callable[[EngineTile, EngineTile], bool],
    ) -> List[PairCandidate]:
        """
        Додає пари вільних плиток, які мають різні імена шаблонів, але візуально збігаються.

        Потрібно для ігор, де два однакові малюнки помилково отримують різний tile_type.
        """
        used: set[tuple[int, int]] = set()
        for p in pairs:
            used.add((min(p.first_id, p.second_id), max(p.first_id, p.second_id)))

        tile_by_id = {t.id: t for t in tiles}
        free_list = self.find_free_tiles(tiles, relations)
        extra: List[PairCandidate] = []

        for i in range(len(free_list)):
            for j in range(i + 1, len(free_list)):
                a = free_list[i]
                b = free_list[j]
                if a.tile_type == b.tile_type:
                    continue
                if (
                    min(a.confidence, b.confidence)
                    < self.pair_min_confidence
                ):
                    continue
                lo, hi = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                if (lo, hi) in used:
                    continue
                ta = tile_by_id[lo]
                tb = tile_by_id[hi]
                if not tiles_look_same(ta, tb):
                    continue
                used.add((lo, hi))
                tt = "|".join(sorted((ta.tile_type, tb.tile_type)))
                pair_id = f"visual:{lo}-{hi}"
                unlock_score = self._estimate_unlock_score((ta, tb), tiles, relations)
                extra.append(
                    PairCandidate(
                        pair_id=pair_id,
                        tile_type=tt,
                        first_id=ta.id,
                        second_id=tb.id,
                        first_coords=(ta.x, ta.y),
                        second_coords=(tb.x, tb.y),
                        first_w=int(ta.w),
                        first_h=int(ta.h),
                        second_w=int(tb.w),
                        second_h=int(tb.h),
                        unlock_score=unlock_score,
                    )
                )

        merged = list(pairs) + extra
        merged.sort(key=lambda p: p.unlock_score, reverse=True)
        return merged

    def ask_lm_studio_for_best_pair(
        self,
        pairs: List[PairCandidate],
        model: str = "qwen/qwen3.5-9b",
        endpoint: str = "http://127.0.0.1:1234/v1/chat/completions",
        timeout_sec: float = 20.0,
    ) -> Dict[str, object]:
        """
        Відправляє список пар у локальну модель LM Studio.

        Очікуваний формат відповіді моделі (JSON):
        {
          "chosen_pair_id": "bamboo_1:12-40",
          "reason": "...",
          "expected_unlocks": 4
        }
        """
        if not pairs:
            return {
                "chosen_pair_id": None,
                "reason": "Немає доступних пар.",
                "expected_unlocks": 0,
                "source": "engine",
            }

        payload_pairs = [
            {
                "pair_id": p.pair_id,
                "type": p.tile_type,
                "first_tile_id": p.first_id,
                "second_tile_id": p.second_id,
                "first_coords": list(p.first_coords),
                "second_coords": list(p.second_coords),
                "first_size": [p.first_w, p.first_h],
                "second_size": [p.second_w, p.second_h],
                "heuristic_unlock_score": p.unlock_score,
            }
            for p in pairs
        ]

        system_prompt = (
            "Ти Mahjong-асистент. Обери ОДНУ найкращу пару для видалення, "
            "щоб максимально розблокувати інші плитки. "
            "Поле chosen_pair_id має збігатися ЗНАЧЕННЯМ з одним із полів pair_id "
            "з вхідного JSON — не вигадуй нові ідентифікатори. "
            "Відповідай тільки JSON без markdown."
        )
        user_prompt = json.dumps(
            {"available_pairs": payload_pairs},
            ensure_ascii=False,
        )

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            content = (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "{}")
            )
            model_json = json.loads(content)
            chosen = model_json.get("chosen_pair_id")
            if chosen:
                model_json["source"] = "lm_studio"
                return model_json
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            # Мережа/парсинг можуть падати — не ламаємо потік гри.
            pass

        # Fallback: локальна евристика з найбільшим unlock_score.
        best = max(pairs, key=lambda p: p.unlock_score)
        return {
            "chosen_pair_id": best.pair_id,
            "reason": "Fallback на локальну евристику (LM Studio недоступний або невалідна відповідь).",
            "expected_unlocks": best.unlock_score,
            "source": "engine_fallback",
        }

    def _estimate_unlock_score(
        self,
        pair: Tuple[EngineTile, EngineTile],
        tiles: List[EngineTile],
        relations: Dict[int, Dict[str, List[int]]],
    ) -> int:
        """
        Оцінка, скільки плиток може розблокуватись після видалення пари.
        Це евристика для сортування кандидатів до/без LM Studio.
        """
        remove_ids = {pair[0].id, pair[1].id}
        unlocks = 0

        for tile in tiles:
            if tile.id in remove_ids:
                continue

            rel = relations[tile.id]
            top_left = [tid for tid in rel["top"] if tid not in remove_ids]
            left_left = [tid for tid in rel["left"] if tid not in remove_ids]
            right_left = [tid for tid in rel["right"] if tid not in remove_ids]

            was_blocked = (len(rel["top"]) > 0) or (
                len(rel["left"]) > 0 and len(rel["right"]) > 0
            )
            will_be_free = (len(top_left) == 0) and (
                (len(left_left) == 0) or (len(right_left) == 0)
            )
            if was_blocked and will_be_free:
                unlocks += 1

        return unlocks

    def _side_rows_aligned(self, tile: EngineTile, other: EngineTile) -> bool:
        """Чи плитки в одному «ряду» по Y (ізометрія ламає строге перетинання прямокутників)."""
        vertical_overlap = self._overlap_1d(
            tile.y, tile.y + tile.h, other.y, other.y + other.h
        )
        mh = float(min(tile.h, other.h))
        if mh <= 0:
            return False
        if vertical_overlap >= mh * self.side_overlap_ratio:
            return True
        center_dy = abs(tile.center_y - other.center_y)
        return vertical_overlap >= mh * self.side_overlap_loose_min and (
            center_dy <= mh * self.side_max_center_dy_ratio
        )

    def _is_left_neighbor(self, tile: EngineTile, other: EngineTile) -> bool:
        if other.center_x >= tile.center_x:
            return False
        if not self._side_rows_aligned(tile, other):
            return False
        horizontal_gap = tile.x - (other.x + other.w)
        max_gap = max(10, int(tile.w * self.side_max_gap_ratio))
        return horizontal_gap <= max_gap

    def _is_right_neighbor(self, tile: EngineTile, other: EngineTile) -> bool:
        if other.center_x <= tile.center_x:
            return False
        if not self._side_rows_aligned(tile, other):
            return False
        horizontal_gap = other.x - (tile.x + tile.w)
        max_gap = max(10, int(tile.w * self.side_max_gap_ratio))
        return horizontal_gap <= max_gap

    def _is_top_blocker(self, tile: EngineTile, other: EngineTile) -> bool:
        """
        Чи плитка `other` лежить шаром вище й перекриває `tile` (блокує зняття).

        Кілька z-зсувів: детектор і перспектива зміщують рамки; один з варіантів має
        спіймати реальне накриття кутом верхньої фішки.
        """
        offsets: Tuple[Tuple[int, int], ...] = (
            self.z_offset,
            (0, 0),
            (-12, -10),
            (12, -10),
            (0, -14),
            (-8, 4),
            (8, 4),
        )
        for zx, zy in offsets:
            if self._is_top_blocker_at_offset(tile, other, (zx, zy)):
                return True
        return False

    def _is_top_blocker_at_offset(
        self,
        tile: EngineTile,
        other: EngineTile,
        z_shift: Tuple[int, int],
    ) -> bool:
        ox = other.x + int(z_shift[0])
        oy = other.y + int(z_shift[1])

        overlap_w = self._overlap_1d(tile.x, tile.x + tile.w, ox, ox + other.w)
        overlap_h = self._overlap_1d(tile.y, tile.y + tile.h, oy, oy + other.h)
        if overlap_w <= 0 or overlap_h <= 0:
            return False

        overlap_area = overlap_w * overlap_h
        tile_area = max(1, tile.w * tile.h)
        other_area = max(1, other.w * other.h)
        ratio_on_tile = overlap_area / float(tile_area)
        ratio_on_other = overlap_area / float(other_area)

        is_above = other.center_y <= tile.center_y + (
            tile.h * self.center_line_tolerance_ratio
        )
        if not is_above:
            return False

        if ratio_on_tile >= self.top_overlap_ratio:
            return True

        partial = (
            ratio_on_tile >= self.top_overlap_partial_min
            and overlap_w >= tile.w * self.top_min_cover_w_ratio
            and overlap_h >= tile.h * self.top_min_cover_h_ratio
        )
        if partial:
            return True

        if (
            ratio_on_other >= max(self.top_overlap_ratio, 0.15)
            and overlap_w >= tile.w * (self.top_min_cover_w_ratio * 0.82)
            and overlap_h >= tile.h * (self.top_min_cover_h_ratio * 0.82)
        ):
            return True

        # Смуга зверху: перетин починається у верхній частині нижньої фішки (кут накриття).
        y_overlap_top = max(tile.y, oy)
        if (
            y_overlap_top <= tile.y + tile.h * 0.46
            and overlap_w >= tile.w * 0.15
            and overlap_h >= tile.h * 0.048
            and ratio_on_tile >= 0.024
        ):
            return True

        return False

    @staticmethod
    def _overlap_1d(a1: int, a2: int, b1: int, b2: int) -> int:
        return max(0, min(a2, b2) - max(a1, b1))
