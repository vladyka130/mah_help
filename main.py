from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox

import cv2
import customtkinter as ctk

from engine import EngineTile, MahjongEngine, PairCandidate
from vision import VisionEngine

if sys.platform == "win32":
    from overlay_layered_win32 import Win32LayeredOverlay, build_overlay_bitmap
else:
    Win32LayeredOverlay = None  # type: ignore[misc, assignment]
    build_overlay_bitmap = None  # type: ignore[misc, assignment]

# Товщина фіолетових рамок навколо кожної валідної пари.
_OVERLAY_STROKE_PURPLE = 4

# Автоклік: мінімальні паузи — розпізнавання має бути «майже миттєвим», затримки лише під анімацію гри.
_AUTO_PAUSE_BETWEEN_TILE_CLICKS_SEC = 0.04
_AUTO_PAUSE_AFTER_PAIR_SEC = 0.12
# Після оновлення статусу в автогра (оверлей можна не малювати — швидше цикл кліку).
_AUTO_PAUSE_AFTER_UI_SEC = 0.01
# У автогра не малювати фіолетові рамки кожного кадру — економія головного потоку й часу перед кліком.
_AUTO_SKIP_OVERLAY_DRAW_IN_AUTO = True

# Редагувані з GUI параметри (float на Vision / Engine). Опис — підказка у вікні.
_PARAM_FLOAT_VISION: list[tuple[str, str]] = [
    (
        "threshold",
        "Детекція: поріг шаблону (0.35–0.92). Нижче — більше фішок і хибних рамок на фоні.",
    ),
    (
        "detection_min_gray_std",
        "Фільтр фону: мін. σ яскравості всередині фішки (вище — менше «пустих» рамок).",
    ),
    (
        "detection_min_laplacian_var",
        "Фільтр фону: мін. варіація Лапласіана (дрібна текстура; допомагає порівняти з однорідним тлом).",
    ),
    (
        "merge_relaxed_delta",
        "Злиття сканів: скільки зняти з порогу в «м’якому» шарі (другий прохід).",
    ),
    (
        "merge_relaxed_floor",
        "Злиття: мінімальний поріг м’якого шару (не нижче цього).",
    ),
    (
        "pair_visual_same_type",
        "Пара (однаковий тип шаблону): мін. схожість сірого ROI центру фішки.",
    ),
    (
        "pair_visual_cross_type",
        "Пара (різні типи): поріг сірого ROI.",
    ),
    (
        "pair_visual_cross_augment",
        "Пара підказки «схожі різні файли»: поріг сірого ROI.",
    ),
    (
        "pair_default_inset_ratio",
        "Відступ від краю ROI для пари (частка ширини). Більше — більше центр малюнка.",
    ),
    (
        "pair_augment_inset_ratio",
        "Той самий відступ для режиму augment (різні імена, один візерунок).",
    ),
    (
        "pair_trust_template_conf",
        "Якщо обидва збіги шаблону ≥ цього — використовується поріг pair_visual_if_trusted.",
    ),
    (
        "pair_visual_if_trusted",
        "Поріг сірого ROI, коли обидва шаблони дуже впевнені.",
    ),
    (
        "pair_skip_roi_same_type_min_conf",
        "Якщо min(score) шаблону ≥ цього й тип однаковий — без ROI (швидко). 1.05 = завжди через ROI.",
    ),
    (
        "pair_edge_min_score",
        "Пара: мінімальний збіг градієнтів (контури символу). Вище — менше плутанини «однаковий фон».",
    ),
    (
        "pair_center_extra_inset",
        "Додатковий відступ для другої перевірки «тільки центр» (відсікає схожі краї, різний центр).",
    ),
    (
        "pair_center_min_score_slack",
        "На скільки послабити поріг сірого для центрового ROI відносно основного.",
    ),
    (
        "pair_corner_width_ratio",
        "Доля ширини фішки — зона верхнього лівого кута (цифра рангу).",
    ),
    (
        "pair_corner_height_ratio",
        "Доля висоти фішки — зона верхнього лівого кута.",
    ),
    (
        "pair_corner_min_norm_score",
        "Мін. збіг matchTemplate на кутах; нижче — різні ранги (2 vs 5), не пара.",
    ),
    (
        "pair_hsv_hist_center_inset",
        "Додатковий inset для кольорової гістограми H–S у центрі (схожі фішки з різним кольором візерунка).",
    ),
    (
        "pair_hsv_hist_min_correl",
        "Мін. кореляція H–S гістограм; нижче — різні фішки (9 vs 7 гачків, червоне vs синє).",
    ),
    (
        "luma_min_cluster_separation",
        "Фільтр затемнення: мін. відстань між кластерами яскравості (якщо фільтр увімкнено).",
    ),
    (
        "luma_max_drop_from_brightest",
        "Макс. різниця яскравості з найсвітлішою детекцією; усе що темніше — схоже на затемнену фішку.",
    ),
]
_PARAM_FLOAT_ENGINE: list[tuple[str, str]] = [
    (
        "pair_min_confidence",
        "Двигун: мін. впевненість шаблону, щоб фішка бралася до пар і геометрії.",
    ),
    (
        "side_max_gap_ratio",
        "Двигун: допустимий зазор між сусідами по горизонталі (частка ширини плитки).",
    ),
]
_PARAM_SWITCH_VISION: list[tuple[str, str]] = [
    (
        "filter_darkened_tiles",
        "Відсіювати затемнені фішки за яскравістю (недоступні для ходу).",
    ),
    (
        "pair_edge_gate_enabled",
        "Вимагати збіг градієнтів для пари (разом із сірим ROI). Вимкни — лише сірий поріг.",
    ),
    (
        "pair_center_gate_enabled",
        "Друга перевірка пари на вужчому центрі (різні «майже однакові» фішки з різницею в центрі).",
    ),
    (
        "pair_corner_gate_enabled",
        "Перевірка верхнього лівого кута (цифра): різні бамбуки з однаковим шаблоном не клікаються як пара.",
    ),
    (
        "pair_hsv_hist_gate_enabled",
        "Перевірка кольору візерунка в центрі (HSV); відсіює плутанину схожих сіток з різними кольорами/кількістю елементів.",
    ),
    (
        "filter_low_texture_detections",
        "Відсіювати збіги на майже рівному фоні (менше фіолетових рамок «в нічому»).",
    ),
]


@dataclass
class AnalysisSnapshot:
    """Результат одного повного аналізу (без Tk)."""

    summary: str
    pairs: list[PairCandidate]
    selected_pair: PairCandidate | None
    lm_response: dict[str, object]
    capture_rect: tuple[int, int, int, int]


def _win32_click_screen_pixel(x: int, y: int) -> None:
    """Один клік ЛКМ у екранних координатах (Windows, DPI-aware разом із mss)."""
    import ctypes

    user32 = ctypes.windll.user32
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.015)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP


class MahjongAssistantApp(ctk.CTk):
    """GUI помічника: Vision -> Engine (евристика Python) -> оверлей. Без LLM / LM Studio."""

    def __init__(self) -> None:
        super().__init__()
        # Вузьке вікно, щоб не перекривати поле гри; детальний стан — у заголовку.
        self._set_window_status("готово")
        self.geometry("230x158")
        self.minsize(210, 140)
        self.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.vision = None
        # Поріг пари трохи нижче за Vision threshold — відсікаємо слабкі збіги (узгоджено з дефолтами vision.py).
        # Трохи м’якші пороги геометрії — частіше визначаються «вільні» фішки на ізометричних полях.
        self.engine = MahjongEngine(
            pair_min_confidence=0.38,
            side_max_gap_ratio=0.50,
        )
        self._is_busy = False
        self._auto_active = False
        self._auto_stop_requested = False
        # Панель «Параметри» (розгортання, щоб вузьке вікно не закривало поле).
        self._params_expanded = False
        self._param_entries: dict[str, ctk.CTkEntry] = {}
        self._param_switches: dict[str, ctk.CTkSwitch] = {}

        # Windows: окреме layered-вікно (UpdateLayeredWindow), не Tk — рамки видно, кліки крізь прозорі пікселі.
        self._win32_overlay: Win32LayeredOverlay | None = None
        self._tk_fallback_overlay: tk.Toplevel | None = None

        self._build_ui()
        self._init_vision()
        self._sync_params_from_engine_to_ui()

        self.bind("<Escape>", lambda _e: self._on_escape_key())
        self._global_f4_hotkeys = None
        # Windows: глобальний F4 (pynput), щоб перемикати автогра з ігрового вікна. Інакше — лише TK F4.
        if sys.platform == "win32":
            self._install_global_f4_hotkey_win32()
        else:
            self.bind("<F4>", lambda _e: self._toggle_auto_play())
        self.protocol("WM_DELETE_WINDOW", self._on_app_close_request)

    def _set_window_status(self, text: str) -> None:
        """Короткий стан у заголовку вікна (форма лише з кнопок)."""
        line = " ".join(text.replace("\n", " ").split())
        if len(line) > 72:
            line = line[:69] + "…"
        self.title(f"Mahjong — {line}")

    def _build_ui(self) -> None:
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True, padx=8, pady=8)

        bw, bh = 190, 30
        self.analyze_btn = ctk.CTkButton(
            container,
            text="Аналізувати",
            command=self.start_analysis,
            width=bw,
            height=bh,
        )
        self.analyze_btn.pack(pady=(0, 5))

        self.clear_btn = ctk.CTkButton(
            container,
            text="Очистити оверлей (Esc)",
            command=self.clear_overlay,
            width=bw,
            height=bh,
        )
        self.clear_btn.pack(pady=(0, 5))

        self.auto_btn = ctk.CTkButton(
            container,
            text="Автогра (F4, навіть у грі)",
            command=self._toggle_auto_play,
            width=bw,
            height=bh,
            fg_color="#2d6a4f",
            hover_color="#1b4332",
        )
        self.auto_btn.pack(pady=(0, 4))

        self._params_toggle_btn = ctk.CTkButton(
            container,
            text="▼ Параметри (детекція / пара)",
            command=self._toggle_params_panel,
            width=bw,
            height=26,
            fg_color="#3d5a80",
            hover_color="#2b4a6d",
        )
        self._params_toggle_btn.pack(pady=(0, 0))

        self._params_scroll = ctk.CTkScrollableFrame(
            container, height=320, label_text="Налаштування (значення — у рядку справа)"
        )
        self._build_params_form(self._params_scroll)
        # Згорнуто: не займає місце, вікно вузьке.
        self._params_scroll.pack_forget()

        self._apply_params_btn = ctk.CTkButton(
            container,
            text="Застосувати параметри",
            command=self._apply_params_from_ui,
            width=bw,
            height=28,
            fg_color="#1b4332",
            hover_color="#081c15",
        )
        self._apply_params_btn.pack_forget()

    def _build_params_form(self, scroll: ctk.CTkScrollableFrame) -> None:
        """Поля числових параметрів та перемикачі (копіювання Ctrl+C/V у Entry працює)."""
        self._param_entries.clear()
        self._param_switches.clear()

        sec_v = ctk.CTkLabel(
            scroll,
            text="Vision (розпізнавання)",
            font=ctk.CTkFont(weight="bold"),
        )
        sec_v.pack(anchor="w", pady=(4, 2))

        for attr, hint in _PARAM_FLOAT_VISION:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(
                row,
                text=f"{attr}\n{hint}",
                anchor="w",
                justify="left",
                font=ctk.CTkFont(size=11),
                wraplength=300,
            ).pack(side="left", fill="x", expand=True, padx=(0, 4))
            ent = ctk.CTkEntry(row, width=76, height=28)
            ent.pack(side="right")
            self._param_entries[f"vision:{attr}"] = ent

        for attr, hint in _PARAM_SWITCH_VISION:
            sw_row = ctk.CTkFrame(scroll, fg_color="transparent")
            sw_row.pack(fill="x", pady=4)
            sw = ctk.CTkSwitch(sw_row, text=f"{attr}: {hint}")
            sw.pack(side="left", anchor="w")
            self._param_switches[f"vision:{attr}"] = sw

        sec_e = ctk.CTkLabel(
            scroll,
            text="Engine (вільні фішки / геометрія)",
            font=ctk.CTkFont(weight="bold"),
        )
        sec_e.pack(anchor="w", pady=(10, 2))

        for attr, hint in _PARAM_FLOAT_ENGINE:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(
                row,
                text=f"{attr}\n{hint}",
                anchor="w",
                justify="left",
                font=ctk.CTkFont(size=11),
                wraplength=300,
            ).pack(side="left", fill="x", expand=True, padx=(0, 4))
            ent = ctk.CTkEntry(row, width=76, height=28)
            ent.pack(side="right")
            self._param_entries[f"engine:{attr}"] = ent

        tip = ctk.CTkLabel(
            scroll,
            text="pair_skip… 1.05 = завжди ROI; ~0.90 = інколи пропуск ROI при дуже високому збігу шаблону. Плутанина пар — підвищ pair_visual_* або edge.",
            font=ctk.CTkFont(size=10),
            text_color="gray70",
            wraplength=310,
        )
        tip.pack(anchor="w", pady=(6, 2))

    def _toggle_params_panel(self) -> None:
        """Показати/сховати блок параметрів і змінити розмір вікна."""
        self._params_expanded = not self._params_expanded
        if self._params_expanded:
            self._params_scroll.pack(fill="both", expand=True, pady=(6, 4))
            self._apply_params_btn.pack(pady=(0, 4))
            self._params_toggle_btn.configure(text="▲ Сховати параметри")
            self.geometry("360x540")
            self.minsize(320, 400)
            self.resizable(True, True)
            self._sync_params_from_engine_to_ui()
        else:
            self._params_scroll.pack_forget()
            self._apply_params_btn.pack_forget()
            self._params_toggle_btn.configure(text="▼ Параметри (детекція / пара)")
            self.geometry("230x158")
            self.minsize(210, 140)
            self.resizable(False, False)

    def _sync_params_from_engine_to_ui(self) -> None:
        """Підставити поточні значення з vision/engine у поля."""
        v = self.vision
        e = self.engine
        for key, w in self._param_entries.items():
            scope, attr = key.split(":", 1)
            try:
                if scope == "vision" and v is not None:
                    w.delete(0, "end")
                    w.insert(0, str(getattr(v, attr)))
                elif scope == "engine":
                    w.delete(0, "end")
                    w.insert(0, str(getattr(e, attr)))
                else:
                    w.delete(0, "end")
                    w.insert(0, "—")
            except Exception:
                w.delete(0, "end")
                w.insert(0, "—")

        for key, sw in self._param_switches.items():
            scope, attr = key.split(":", 1)
            if scope == "vision" and v is not None:
                if getattr(v, attr):
                    sw.select()
                else:
                    sw.deselect()
            else:
                sw.deselect()

    def _apply_params_from_ui(self) -> None:
        """Записати числа з полів у VisionEngine / MahjongEngine (без перезавантаження шаблонів)."""
        if self.vision is None:
            messagebox.showwarning("Параметри", "Vision ще не ініціалізовано.")
            return
        v = self.vision
        e = self.engine
        try:
            for key, w in self._param_entries.items():
                scope, attr = key.split(":", 1)
                text = w.get().strip().replace(",", ".")
                if not text or text == "—":
                    continue
                val = float(text)
                val = self._clamp_runtime_param(scope, attr, val)
                if scope == "vision":
                    setattr(v, attr, val)
                else:
                    setattr(e, attr, val)

            for key, sw in self._param_switches.items():
                scope, attr = key.split(":", 1)
                if scope == "vision":
                    setattr(v, attr, self._switch_is_on(sw))
        except ValueError as exc:
            messagebox.showerror("Параметри", f"Перевірте числа в полях.\n{exc}")
            return

        self._sync_params_from_engine_to_ui()
        self._set_window_status("Параметри застосовано")
        messagebox.showinfo(
            "Параметри",
            "Значення записано (числа обмежені безпечним діапазоном). Наступний аналіз — нові пороги.",
        )

    @staticmethod
    def _clamp_runtime_param(scope: str, attr: str, val: float) -> float:
        """Обмежити екстремальні значення з полів (щоб не «обнулити» детекцію випадковим числом)."""
        x = float(val)
        if scope == "vision":
            if attr == "threshold":
                return max(0.30, min(0.92, x))
            if attr == "merge_relaxed_floor":
                return max(0.22, min(0.58, x))
            if attr == "merge_relaxed_delta":
                return max(0.05, min(0.35, x))
            if attr == "pair_edge_min_score":
                return max(0.18, min(0.72, x))
            if attr.startswith("pair_visual") or attr in (
                "pair_trust_template_conf",
                "pair_visual_if_trusted",
            ):
                return max(0.28, min(0.85, x))
            if attr.endswith("inset_ratio"):
                return max(0.02, min(0.45, x))
            if attr == "pair_skip_roi_same_type_min_conf":
                return max(0.35, min(1.08, x))
            if attr == "detection_min_gray_std":
                return max(6.0, min(28.0, x))
            if attr == "detection_min_laplacian_var":
                return max(2.0, min(80.0, x))
            if attr == "luma_min_cluster_separation":
                return max(3.0, min(25.0, x))
            if attr == "luma_max_drop_from_brightest":
                return max(12.0, min(65.0, x))
            if attr == "pair_center_extra_inset":
                return max(0.02, min(0.25, x))
            if attr == "pair_center_min_score_slack":
                return max(0.01, min(0.22, x))
            if attr in ("pair_corner_width_ratio", "pair_corner_height_ratio"):
                return max(0.12, min(0.50, x))
            if attr == "pair_corner_min_norm_score":
                return max(0.40, min(0.95, x))
            if attr == "pair_hsv_hist_center_inset":
                return max(0.04, min(0.30, x))
            if attr == "pair_hsv_hist_min_correl":
                return max(0.30, min(0.95, x))
        if scope == "engine":
            if attr == "pair_min_confidence":
                return max(0.28, min(0.88, x))
            if attr == "side_max_gap_ratio":
                return max(0.25, min(0.75, x))
        return x

    @staticmethod
    def _switch_is_on(sw: ctk.CTkSwitch) -> bool:
        """Стан CTkSwitch у bool (різні версії CustomTkinter дають 1/0 або «on»/«off»)."""
        g = sw.get()
        if isinstance(g, str):
            return g.lower() in ("on", "1", "true", "yes")
        return bool(g)

    def _init_vision(self) -> None:
        # Усі «робочі» пороги задані в vision.VisionEngine.__init__ (дефолти): панель параметрів не обов’язкова.
        try:
            self.vision = VisionEngine(templates_dir="assets/tiles")
            self._set_window_status(
                "Vision OK; затемнені фішки відсікаються за яскравістю"
            )
        except Exception as exc:
            self._set_window_status("Помилка Vision / шаблонів")
            messagebox.showerror("Помилка", f"Не вдалося завантажити шаблони:\n{exc}")

    def start_analysis(self) -> None:
        if self._auto_active:
            messagebox.showinfo("Авто", "Спочатку зупиніть автогра (кнопка або Esc).")
            return
        if self._is_busy:
            return
        if self.vision is None:
            messagebox.showwarning("Увага", "Vision не ініціалізовано.")
            return

        self._is_busy = True
        self.analyze_btn.configure(state="disabled")
        self._set_window_status("Аналіз…")
        threading.Thread(target=self._run_analysis_task, daemon=True).start()

    def _pair_visual_min_for_tiles(self, a: EngineTile, b: EngineTile) -> float:
        """Поріг схожості ROI для пари плиток (адаптивно до типу й впевненості шаблону)."""
        v = self.vision
        assert v is not None
        if a.tile_type != b.tile_type:
            return v.pair_visual_cross_type
        if min(a.confidence, b.confidence) >= v.pair_trust_template_conf:
            return v.pair_visual_if_trusted
        return v.pair_visual_same_type

    def _pair_visual_min_for_candidate(
        self, p: PairCandidate, tile_by_id: dict[int, EngineTile]
    ) -> float:
        """Поріг для PairCandidate після find_free_pairs / augment."""
        assert self.vision is not None
        if "|" in p.tile_type or p.pair_id.startswith("visual:"):
            return self.vision.pair_visual_cross_type
        ta = tile_by_id[p.first_id]
        tb = tile_by_id[p.second_id]
        return self._pair_visual_min_for_tiles(ta, tb)

    def _analyze_once_core(self, *, for_auto: bool = False) -> AnalysisSnapshot:
        """Повний аналіз у робочому потоці (без викликів Tk). ``for_auto`` — швидкий шлях для автогра."""
        assert self.vision is not None
        v = self.vision
        _frame, cap_rect = v.capture_screen(region=None)
        reuse_scale = bool(for_auto)
        # Один базовий прохід find_templates — стабільніший за злиття двох сканів на проблемних DPI.
        matches, grouped = v.find_templates(
            _frame, reuse_cached_scale=reuse_scale
        )
        snap = self._snapshot_from_detection(
            _frame,
            matches,
            grouped,
            cap_rect,
            relaxed_suffix="",
            turbo=for_auto,
        )
        # Другий / третій прохід — навіть якщо перший дав 0 детекцій (умова лише «нема пар»).
        if len(snap.pairs) == 0:
            th2 = max(0.40, float(v.threshold) - 0.11)
            matches2, grouped2 = v.find_templates(
                _frame,
                threshold=th2,
                apply_darkened_filter=None,
                reuse_cached_scale=reuse_scale,
            )
            snap2 = self._snapshot_from_detection(
                _frame,
                matches2,
                grouped2,
                cap_rect,
                relaxed_suffix=" | Другий прохід (м’якший поріг шаблону)",
                turbo=for_auto,
            )
            if len(snap2.pairs) > len(snap.pairs) or len(matches2) > len(matches):
                snap, matches, grouped = snap2, matches2, grouped2

        # У автогра без третього проходу (економія ~цілого matchTemplate-циклу); ручний аналіз — повний відкат.
        if len(snap.pairs) == 0 and not for_auto:
            th3 = max(0.42, float(v.threshold) - 0.14)
            matches3, grouped3 = v.find_templates(
                _frame,
                threshold=th3,
                apply_darkened_filter=None,
                reuse_cached_scale=reuse_scale,
            )
            snap3 = self._snapshot_from_detection(
                _frame,
                matches3,
                grouped3,
                cap_rect,
                relaxed_suffix=" | Третій прохід (макс. чутливість)",
                turbo=for_auto,
            )
            if len(snap3.pairs) > len(snap.pairs) or len(matches3) > len(matches):
                snap, matches, grouped = snap3, matches3, grouped3

        return snap

    def _snapshot_from_detection(
        self,
        _frame,
        matches,
        grouped,
        cap_rect,
        relaxed_suffix: str,
        *,
        turbo: bool = False,
    ) -> AnalysisSnapshot:
        """Збирає пари та підказку з уже знайдених детекцій. ``turbo`` — прискорені перевірки ROI в автогра."""
        tiles = self.engine.build_tiles(matches)
        tile_by_id = {t.id: t for t in tiles}
        relations = self.engine.build_relations(tiles)
        free_pairs_raw = self.engine.find_free_pairs(tiles, relations)
        pairs_before_visual = len(free_pairs_raw)
        free_pairs: list[PairCandidate] = []
        v = self.vision
        assert v is not None
        # Один перехід BGR→сірий на весь кадр: інакше кожна перевірка пари робить повний cvtColor.
        frame_gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
        for p in free_pairs_raw:
            ta = tile_by_id[p.first_id]
            tb = tile_by_id[p.second_id]
            same_named_type = (
                "|" not in p.tile_type
                and not str(p.pair_id).startswith("visual:")
                and ta.tile_type == tb.tile_type
            )
            # Бамбук 2 vs 5: навіть якщо шаблон дав один тип — верхній лівий кут (цифра) має збігатися.
            if same_named_type and v.pair_corner_gate_enabled:
                if not v.pair_top_left_corners_look_same(
                    frame_gray,
                    ta.x,
                    ta.y,
                    ta.w,
                    ta.h,
                    tb.x,
                    tb.y,
                    tb.w,
                    tb.h,
                ):
                    continue
            # Довіра до шаблону для однакового типу — як у простішій збірці (менше «відсіяно за зображенням»).
            skip_roi = same_named_type and min(ta.confidence, tb.confidence) >= float(
                v.pair_skip_roi_same_type_min_conf
            )
            if skip_roi:
                free_pairs.append(p)
                continue
            if v.pair_patches_look_same(
                _frame,
                p.first_coords[0],
                p.first_coords[1],
                p.first_w,
                p.first_h,
                p.second_coords[0],
                p.second_coords[1],
                p.second_w,
                p.second_h,
                inset_ratio=v.pair_default_inset_ratio,
                min_normalized_score=self._pair_visual_min_for_candidate(
                    p, tile_by_id
                ),
                frame_gray=frame_gray,
                fast=turbo,
            ):
                free_pairs.append(p)
        pairs_visual_dropped = pairs_before_visual - len(free_pairs)

        # Різні імена шаблонів при однаковому малюнку — augment це O(k²) візуальних викликів;
        # у turbo пропускаємо, якщо вже є хоча б одна валідна пара (типовий випадок ~100 фішок).
        if not (turbo and len(free_pairs) > 0):

            def _augment_visual_same(a: EngineTile, b: EngineTile) -> bool:
                return bool(
                    v.pair_patches_look_same(
                        _frame,
                        a.x,
                        a.y,
                        a.w,
                        a.h,
                        b.x,
                        b.y,
                        b.w,
                        b.h,
                        inset_ratio=v.pair_augment_inset_ratio,
                        min_normalized_score=v.pair_visual_cross_augment,
                        frame_gray=frame_gray,
                        fast=turbo,
                    )
                )

            free_pairs = self.engine.augment_free_pairs_cross_type_visual(
                free_pairs,
                tiles,
                relations,
                _augment_visual_same,
            )

        # Вибір «найкращої» пари лише в Python: максимальний unlock_score у MahjongEngine.
        if not free_pairs:
            lm_response = {
                "chosen_pair_id": None,
                "reason": "Немає доступних пар.",
                "expected_unlocks": 0,
                "source": "python_heuristic",
            }
        else:
            best = max(free_pairs, key=lambda p: p.unlock_score)
            lm_response = {
                "chosen_pair_id": best.pair_id,
                "reason": "Найкраща пара за евристикою unlock_score (Python, без LLM).",
                "expected_unlocks": best.unlock_score,
                "source": "python_heuristic",
            }

        selected_pair = self._resolve_selected_pair(free_pairs, lm_response)
        summary = (
            f"Плиток: {len(matches)} | Типів: {len(grouped)} | "
            f"Доступних пар: {len(free_pairs)}"
        )
        if pairs_visual_dropped > 0:
            summary += f" | Відсіяно за зображенням: {pairs_visual_dropped}"
        if relaxed_suffix:
            summary += relaxed_suffix
        return AnalysisSnapshot(
            summary=summary,
            pairs=list(free_pairs),
            selected_pair=selected_pair,
            lm_response=lm_response,
            capture_rect=cap_rect,
        )

    def _run_analysis_task(self) -> None:
        try:
            snap = self._analyze_once_core()
            self.after(0, lambda: self._finish_pipeline_from_snapshot(snap))
        except Exception as exc:
            self.after(0, lambda e=str(exc): self._finish_analysis("", e))

    def _finish_pipeline_from_snapshot(self, snap: AnalysisSnapshot) -> None:
        self._is_busy = False
        self.analyze_btn.configure(state="normal")
        self._set_window_status(snap.summary)
        self._draw_overlay(
            snap.pairs,
            snap.selected_pair,
            snap.lm_response,
            snap.capture_rect,
        )

    def _apply_auto_snapshot(
        self, snap: AnalysisSnapshot, cycle_step: int = 0
    ) -> None:
        """Оновлення статусу й оверлею під час автогра (кнопка аналізу лишається вимкненою)."""
        if cycle_step > 0:
            extra = f" | Автогра, крок {cycle_step}…"
        else:
            extra = " | Автогра…"
        self._set_window_status(snap.summary + extra)
        if not (_AUTO_SKIP_OVERLAY_DRAW_IN_AUTO and self._auto_active):
            self._draw_overlay(
                snap.pairs,
                snap.selected_pair,
                snap.lm_response,
                snap.capture_rect,
            )

    def _click_pair_centers_screen(
        self, pair: PairCandidate, cap_left: int, cap_top: int
    ) -> None:
        """
        Дві послідовні ЛКМ: спочатку по центру першої плитки пари, потім по центру другої.

        Пара вже відібрана як дві однакові за правилами гри (тип + вільність + візуальний збіг);
        це не «будь-які дві фішки», а саме рекомендована пара однакових розкритих плиток.
        """
        x1 = cap_left + pair.first_coords[0] + pair.first_w // 2
        y1 = cap_top + pair.first_coords[1] + pair.first_h // 2
        x2 = cap_left + pair.second_coords[0] + pair.second_w // 2
        y2 = cap_top + pair.second_coords[1] + pair.second_h // 2
        _win32_click_screen_pixel(x1, y1)
        time.sleep(_AUTO_PAUSE_BETWEEN_TILE_CLICKS_SEC)
        _win32_click_screen_pixel(x2, y2)

    def _toggle_auto_play(self) -> None:
        if self.vision is None:
            messagebox.showwarning("Увага", "Vision не ініціалізовано.")
            return
        if self._auto_active:
            self._auto_stop_requested = True
            self._set_window_status("Зупинка авто…")
            return
        if self._is_busy:
            return
        if sys.platform != "win32":
            messagebox.showwarning(
                "Автоклік",
                "Автоматичні кліки підтримуються лише на Windows.",
            )
            return
        self._auto_stop_requested = False
        self._auto_active = True
        self._is_busy = True
        self.analyze_btn.configure(state="disabled")
        self.auto_btn.configure(
            text="Зупинити авто",
            fg_color="#9d0208",
            hover_color="#6a040f",
        )
        threading.Thread(target=self._auto_play_worker, daemon=True).start()

    def _auto_play_worker(self) -> None:
        try:
            cycle_step = 0
            while not self._auto_stop_requested:
                snap = self._analyze_once_core(for_auto=True)
                if self._auto_stop_requested:
                    break
                if not snap.pairs or snap.selected_pair is None:
                    self.after(
                        0,
                        lambda: self._auto_play_finished_msg(
                            "Авто: немає доступних пар — зупинка."
                        ),
                    )
                    break
                cycle_step += 1
                self.after(
                    0,
                    lambda s=snap, st=cycle_step: self._apply_auto_snapshot(s, st),
                )
                time.sleep(_AUTO_PAUSE_AFTER_UI_SEC)
                if self._auto_stop_requested:
                    break
                # Зняти оверлей перед кліком: інакше topmost-вікно може перехопити ЛКМ замість гри.
                self._wait_hide_overlay_for_auto_click()
                if self._auto_stop_requested:
                    break
                cl, ct = snap.capture_rect[0], snap.capture_rect[1]
                self._click_pair_centers_screen(snap.selected_pair, cl, ct)
                time.sleep(_AUTO_PAUSE_AFTER_PAIR_SEC)
        except Exception as exc:
            self.after(
                0,
                lambda e=str(exc): messagebox.showerror("Помилка автогра", e),
            )
        finally:
            self.after(0, self._auto_play_cleanup_ui)

    def _auto_play_finished_msg(self, msg: str) -> None:
        self._set_window_status(msg)
        self.clear_overlay()

    def _auto_play_cleanup_ui(self) -> None:
        self._auto_active = False
        self._auto_stop_requested = False
        self._is_busy = False
        self.analyze_btn.configure(state="normal")
        self.auto_btn.configure(
            text="Автогра (F4, навіть у грі)",
            fg_color="#2d6a4f",
            hover_color="#1b4332",
        )

    def _on_escape_key(self) -> None:
        self.clear_overlay()

    def _install_global_f4_hotkey_win32(self) -> None:
        """
        Глобальна гаряча клавіша F4 (працює при фокусі на грі).
        Потрібен пакет pynput; інакше лишається лише F4, коли активне вікно помічника.
        """

        def bind_f4_local() -> None:
            self.bind("<F4>", lambda _e: self._toggle_auto_play())

        def on_global_f4() -> None:
            try:
                self.after(0, self._toggle_auto_play)
            except Exception:
                pass

        try:
            from pynput.keyboard import GlobalHotKeys
        except ImportError:
            bind_f4_local()
            self.after(
                200,
                lambda: self._set_window_status(
                    "Глобальний F4: pip install pynput (поки лише F4 у вікні програми)"
                ),
            )
            return

        def run_hotkey_thread() -> None:
            try:
                hk = GlobalHotKeys({"<f4>": on_global_f4})
                self._global_f4_hotkeys = hk
                hk.start()
            except Exception:
                self.after(0, bind_f4_local)
                self.after(
                    200,
                    lambda: self._set_window_status(
                        "F4 глобально не вдалось — використовуйте F4 у вікні помічника"
                    ),
                )

        threading.Thread(target=run_hotkey_thread, daemon=True).start()

    def _on_app_close_request(self) -> None:
        """Зупинка слухача F4 і закриття вікна."""
        self._auto_stop_requested = True
        hk = getattr(self, "_global_f4_hotkeys", None)
        if hk is not None:
            try:
                hk.stop()
            except Exception:
                pass
            self._global_f4_hotkeys = None
        self.quit()
        self.destroy()

    def _finish_analysis(self, message: str, error: str | None) -> None:
        self._is_busy = False
        self.analyze_btn.configure(state="normal")
        if error:
            self._set_window_status("Помилка аналізу")
            messagebox.showerror("Помилка аналізу", error)
            return
        self._set_window_status(message)

    def _resolve_selected_pair(
        self, pairs: list[PairCandidate], lm_response: dict[str, object]
    ) -> PairCandidate | None:
        if not pairs:
            return None
        chosen_id = str(lm_response.get("chosen_pair_id") or "").strip()
        for pair in pairs:
            if pair.pair_id == chosen_id:
                return pair
        return max(pairs, key=lambda p: p.unlock_score)

    def _draw_overlay(
        self,
        pairs: list[PairCandidate],
        selected_pair: PairCandidate | None,
        lm_response: dict[str, object],
        capture_rect: tuple[int, int, int, int],
    ) -> None:
        # Лише прибрати старий оверлей. Не викликати clear_overlay() — там вмикається
        # зупинка автогра; інакше цикл «пошук → клік» обривався б після першого кроку.
        self._destroy_overlay_visual_only()
        if not pairs:
            return

        cap_left, cap_top, cap_w, cap_h = capture_rect
        if (
            sys.platform == "win32"
            and Win32LayeredOverlay is not None
            and build_overlay_bitmap is not None
        ):
            try:
                bmp = build_overlay_bitmap(
                    cap_w,
                    cap_h,
                    pairs,
                    selected_pair,
                    lm_response,
                    _OVERLAY_STROKE_PURPLE,
                )
                self._win32_overlay = Win32LayeredOverlay()
                self._win32_overlay.show_bitmap(
                    cap_left, cap_top, cap_w, cap_h, bmp
                )
            except OSError as exc:
                messagebox.showerror(
                    "Оверлей",
                    f"Не вдалося показати шаровий оверлей:\n{exc}",
                )
            return

        # Резерв (не Windows): простий Tk без passthrough.
        self._draw_overlay_tk_fallback(
            pairs, selected_pair, lm_response, capture_rect
        )

    def _draw_overlay_tk_fallback(
        self,
        pairs: list[PairCandidate],
        selected_pair: PairCandidate | None,
        lm_response: dict[str, object],
        capture_rect: tuple[int, int, int, int],
    ) -> None:
        cap_left, cap_top, cap_w, cap_h = capture_rect
        top = tk.Toplevel(self)
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        top.attributes("-alpha", 0.70)
        top.configure(bg="black")
        top.geometry(f"{cap_w}x{cap_h}+{cap_left}+{cap_top}")
        cvs = tk.Canvas(
            top, width=cap_w, height=cap_h, bg="black", highlightthickness=0
        )
        cvs.pack(fill="both", expand=True)
        for pair in pairs:
            self._draw_pair_on_canvas(
                cvs, pair, "#bb33ff", _OVERLAY_STROKE_PURPLE
            )
        if selected_pair:
            expected_unlocks = max(
                0,
                int(lm_response.get("expected_unlocks", selected_pair.unlock_score)),
            )
            tip_text = f"Рекомендовано: ~{expected_unlocks} плиток"
            tx = min(
                selected_pair.first_coords[0], selected_pair.second_coords[0]
            )
            ty = max(
                15,
                min(
                    selected_pair.first_coords[1],
                    selected_pair.second_coords[1],
                )
                - 18,
            )
            cvs.create_text(
                tx,
                ty,
                text=tip_text,
                fill="#ffd54f",
                anchor="nw",
                font=("Segoe UI", 11, "bold"),
            )
        self._tk_fallback_overlay = top

    def _draw_pair_on_canvas(
        self,
        canvas: tk.Canvas,
        pair: PairCandidate,
        color: str,
        width: int,
    ) -> None:
        x1, y1 = pair.first_coords
        x2, y2 = pair.second_coords
        w1, h1 = pair.first_w, pair.first_h
        w2, h2 = pair.second_w, pair.second_h
        canvas.create_rectangle(
            x1, y1, x1 + w1, y1 + h1, outline=color, width=width
        )
        canvas.create_rectangle(
            x2, y2, x2 + w2, y2 + h2, outline=color, width=width
        )

    def _destroy_overlay_visual_only(self) -> None:
        """Закрити лише вікна оверлею (без зміни прапорців автогра). Викликати з головного потоку Tk."""
        if self._win32_overlay is not None:
            try:
                self._win32_overlay.destroy()
            except Exception:
                pass
            self._win32_overlay = None
        tf = getattr(self, "_tk_fallback_overlay", None)
        if tf is not None:
            try:
                tf.destroy()
            except tk.TclError:
                pass
            self._tk_fallback_overlay = None

    def _wait_hide_overlay_for_auto_click(self) -> None:
        """
        Фоновий потік авто чекає, поки головний потік прибере оверлей — щоб клік потрапив у гру.
        """
        ready = threading.Event()

        def _on_main() -> None:
            self._destroy_overlay_visual_only()
            ready.set()

        self.after(0, _on_main)
        ready.wait(timeout=3.0)

    def clear_overlay(self) -> None:
        # Зупиняє автогра, якщо вона була активна (Esc / «Очистити»).
        self._auto_stop_requested = True
        self._destroy_overlay_visual_only()


def _try_set_dpi_awareness() -> None:
    """Вирівнює координати Tk з mss при масштабуванні Windows (125% тощо)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main() -> None:
    _try_set_dpi_awareness()
    app = MahjongAssistantApp()
    app.mainloop()


if __name__ == "__main__":
    main()
