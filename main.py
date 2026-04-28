from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox

import customtkinter as ctk

from engine import MahjongEngine, PairCandidate
from vision import VisionEngine

if sys.platform == "win32":
    from overlay_layered_win32 import Win32LayeredOverlay, build_overlay_bitmap
else:
    Win32LayeredOverlay = None  # type: ignore[misc, assignment]
    build_overlay_bitmap = None  # type: ignore[misc, assignment]

# Товщина фіолетових рамок навколо кожної валідної пари.
_OVERLAY_STROKE_PURPLE = 4

# Автоклік: пауза між двома плитками пари та після пари (анімація гри).
_AUTO_PAUSE_BETWEEN_TILE_CLICKS_SEC = 0.18
_AUTO_PAUSE_AFTER_PAIR_SEC = 0.72
# Коротка пауза після оновлення оверлею, щоб головний потік встиг намалювати.
_AUTO_PAUSE_AFTER_UI_SEC = 0.12


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
    time.sleep(0.04)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP


class MahjongAssistantApp(ctk.CTk):
    """GUI помічника: Vision -> Engine -> LM Studio -> Overlay."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Mahjong Assistant")
        self.geometry("500x400")
        self.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.vision = None
        # Поріг пари трохи нижче за Vision threshold — відсікаємо слабкі збіги ще до LM.
        self.engine = MahjongEngine(pair_min_confidence=0.56)
        self._is_busy = False
        self._auto_active = False
        self._auto_stop_requested = False
        self.status_var = ctk.StringVar(value="Готово до аналізу.")
        # У циклі авто LM викликає HTTP щоразу — повільно; вимкни для швидшої гри.
        self._auto_use_lm_var = ctk.BooleanVar(value=False)

        # Windows: окреме layered-вікно (UpdateLayeredWindow), не Tk — рамки видно, кліки крізь прозорі пікселі.
        self._win32_overlay: Win32LayeredOverlay | None = None
        self._tk_fallback_overlay: tk.Toplevel | None = None

        self._build_ui()
        self._init_vision()

        self.bind("<Escape>", lambda _e: self._on_escape_key())
        self.bind("<F4>", lambda _e: self._on_escape_key())

    def _build_ui(self) -> None:
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True, padx=14, pady=14)

        title = ctk.CTkLabel(
            container,
            text="Помічник Mahjong Solitaire",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title.pack(pady=(10, 8))

        ctk.CTkLabel(container, textvariable=self.status_var).pack(pady=(0, 12))

        self.analyze_btn = ctk.CTkButton(
            container,
            text="Аналізувати",
            command=self.start_analysis,
            width=210,
            height=40,
        )
        self.analyze_btn.pack(pady=(0, 6))

        self.clear_btn = ctk.CTkButton(
            container,
            text="Очистити (Esc/F4)",
            command=self.clear_overlay,
            width=210,
            height=34,
        )
        self.clear_btn.pack(pady=(0, 8))

        self.auto_btn = ctk.CTkButton(
            container,
            text="Автогра (клік + аналіз циклом)",
            command=self._toggle_auto_play,
            width=210,
            height=36,
            fg_color="#2d6a4f",
            hover_color="#1b4332",
        )
        self.auto_btn.pack(pady=(0, 6))

        ctk.CTkCheckBox(
            container,
            text="У авто питати LM Studio (повільніше; інакше лише евристика пар)",
            variable=self._auto_use_lm_var,
        ).pack(pady=(0, 8))

        ctk.CTkLabel(
            container,
            text=(
                "Час аналізу: основне — matchTemplate по шаблонах; підбір масштабу прискорено "
                "(прев’ю кадру, грубо+тонко, повторне використання масштабу між кадрами).\n"
                "Фіолетові рамки — пари; жовтий текст — рекомендація. Esc/F4 — зупинити авто й прибрати оверлей."
            ),
            wraplength=460,
            justify="left",
        ).pack(pady=(0, 6))

    def _init_vision(self) -> None:
        try:
            self.vision = VisionEngine(
                templates_dir="assets/tiles",
                threshold=0.60,
                template_scale=1.0,
                auto_scale=True,
                auto_scale_range=(0.60, 1.45),
                auto_scale_step=0.05,
                # Швидший підбір масштабу: прев’ю кадр, грубо+тонко, кеш попереднього s.
                auto_scale_coarse_step=0.09,
                auto_scale_fine_step=0.03,
                auto_scale_preview_max_side=1280,
                auto_scale_sample_templates=6,
            )
            self.status_var.set(
                "Vision: threshold=0.60, швидкий підбір масштабу (прев’ю + кеш)."
            )
        except Exception as exc:
            self.status_var.set("Помилка ініціалізації Vision.")
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
        self.status_var.set("Виконую: Screenshot -> Vision -> Engine -> LM Studio...")
        threading.Thread(target=self._run_analysis_task, daemon=True).start()

    def _analyze_once_core(self, use_lm: bool) -> AnalysisSnapshot:
        """Повний аналіз у робочому потоці (без викликів Tk)."""
        assert self.vision is not None
        _frame, matches, grouped, cap_rect = self.vision.analyze_once(region=None)
        tiles = self.engine.build_tiles(matches)
        relations = self.engine.build_relations(tiles)
        free_pairs = self.engine.find_free_pairs(tiles, relations)
        pairs_before_visual = len(free_pairs)
        free_pairs = [
            p
            for p in free_pairs
            if self.vision.pair_patches_look_same(
                _frame,
                p.first_coords[0],
                p.first_coords[1],
                p.first_w,
                p.first_h,
                p.second_coords[0],
                p.second_coords[1],
                p.second_w,
                p.second_h,
            )
        ]
        pairs_visual_dropped = pairs_before_visual - len(free_pairs)

        free_pairs = self.engine.augment_free_pairs_cross_type_visual(
            free_pairs,
            tiles,
            relations,
            lambda a, b: self.vision.pair_patches_look_same(
                _frame,
                a.x,
                a.y,
                a.w,
                a.h,
                b.x,
                b.y,
                b.w,
                b.h,
            ),
        )

        if use_lm:
            lm_response = dict(self.engine.ask_lm_studio_for_best_pair(free_pairs))
            print("\n[LM Studio]", lm_response, flush=True)
        else:
            if not free_pairs:
                lm_response = {
                    "chosen_pair_id": None,
                    "reason": "Немає доступних пар.",
                    "expected_unlocks": 0,
                    "source": "auto_heuristic",
                }
            else:
                best = max(free_pairs, key=lambda p: p.unlock_score)
                lm_response = {
                    "chosen_pair_id": best.pair_id,
                    "reason": "Авто без LM — найкраща пара за евристикою.",
                    "expected_unlocks": best.unlock_score,
                    "source": "auto_heuristic",
                }

        selected_pair = self._resolve_selected_pair(free_pairs, lm_response)
        summary = (
            f"Плиток: {len(matches)} | Типів: {len(grouped)} | "
            f"Доступних пар: {len(free_pairs)}"
        )
        if pairs_visual_dropped > 0:
            summary += f" | Відсіяно за зображенням: {pairs_visual_dropped}"
        return AnalysisSnapshot(
            summary=summary,
            pairs=list(free_pairs),
            selected_pair=selected_pair,
            lm_response=lm_response,
            capture_rect=cap_rect,
        )

    def _run_analysis_task(self) -> None:
        try:
            snap = self._analyze_once_core(use_lm=True)
            self.after(0, lambda: self._finish_pipeline_from_snapshot(snap))
        except Exception as exc:
            self.after(0, lambda e=str(exc): self._finish_analysis("", e))

    def _finish_pipeline_from_snapshot(self, snap: AnalysisSnapshot) -> None:
        self._is_busy = False
        self.analyze_btn.configure(state="normal")
        self.status_var.set(snap.summary)
        self._draw_overlay(
            snap.pairs,
            snap.selected_pair,
            snap.lm_response,
            snap.capture_rect,
        )

    def _apply_auto_snapshot(self, snap: AnalysisSnapshot) -> None:
        """Оновлення статусу й оверлею під час автогра (кнопка аналізу лишається вимкненою)."""
        extra = " | Автогра…"
        self.status_var.set(snap.summary + extra)
        self._draw_overlay(
            snap.pairs,
            snap.selected_pair,
            snap.lm_response,
            snap.capture_rect,
        )

    def _click_pair_centers_screen(
        self, pair: PairCandidate, cap_left: int, cap_top: int
    ) -> None:
        """Клік по центрах обох плиток пари в екранних координатах."""
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
            self.status_var.set("Зупинка автогра…")
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
        # Знімок опції LM на старті циклу (читання BooleanVar лише з головного потоку Tk).
        self._auto_lm_flag = bool(self._auto_use_lm_var.get())
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
            while not self._auto_stop_requested:
                snap = self._analyze_once_core(use_lm=self._auto_lm_flag)
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
                self.after(0, lambda s=snap: self._apply_auto_snapshot(s))
                time.sleep(_AUTO_PAUSE_AFTER_UI_SEC)
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
        self.status_var.set(msg)
        self.clear_overlay()

    def _auto_play_cleanup_ui(self) -> None:
        self._auto_active = False
        self._auto_stop_requested = False
        self._is_busy = False
        self.analyze_btn.configure(state="normal")
        self.auto_btn.configure(
            text="Автогра (клік + аналіз циклом)",
            fg_color="#2d6a4f",
            hover_color="#1b4332",
        )

    def _on_escape_key(self) -> None:
        self.clear_overlay()

    def _finish_analysis(self, message: str, error: str | None) -> None:
        self._is_busy = False
        self.analyze_btn.configure(state="normal")
        if error:
            self.status_var.set("Помилка під час аналізу.")
            messagebox.showerror("Помилка аналізу", error)
            return
        self.status_var.set(message)

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
        self.clear_overlay()
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
            expected_unlocks = int(
                lm_response.get("expected_unlocks", selected_pair.unlock_score)
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

    def clear_overlay(self) -> None:
        # Зупиняє автогра, якщо вона була активна (Esc / «Очистити»).
        self._auto_stop_requested = True
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
