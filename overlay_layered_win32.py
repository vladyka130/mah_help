"""
Пер-піксельний прозорий оверлей Windows через UpdateLayeredWindow.

Чому не Tk + WS_EX_TRANSPARENT: на дочірніх HWND це ламає малювання Canvas.
Тут бітмап з premultiplied BGRA: де alpha=0 — клік проходить у вікно під оверлеєм.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine import PairCandidate


def premultiply_bgra(straight: np.ndarray) -> np.ndarray:
    """Переводить straight BGRA у premultiplied BGRA для UpdateLayeredWindow."""
    out = np.zeros_like(straight)
    a = straight[:, :, 3].astype(np.float32) / 255.0
    for c in range(3):
        out[:, :, c] = np.clip(
            straight[:, :, c].astype(np.float32) * a, 0, 255
        ).astype(np.uint8)
    out[:, :, 3] = straight[:, :, 3]
    return out


def build_overlay_bitmap(
    cap_w: int,
    cap_h: int,
    pairs: list[PairCandidate],
    selected_pair: PairCandidate | None,
    lm_response: dict[str, object],
    stroke_px: int = 4,
) -> np.ndarray:
    """
    Лише фіолетові рамки навколо кожної валідної пари + текст рекомендації (якщо є).
    """
    img = Image.new("RGBA", (cap_w, cap_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    purple = (187, 51, 255, 245)
    sw = max(1, stroke_px)

    for pair in pairs:
        x1, y1 = pair.first_coords
        x2, y2 = pair.second_coords
        draw.rectangle(
            [x1, y1, x1 + pair.first_w, y1 + pair.first_h],
            outline=purple,
            width=sw,
        )
        draw.rectangle(
            [x2, y2, x2 + pair.second_w, y2 + pair.second_h],
            outline=purple,
            width=sw,
        )

    if selected_pair:
        expected_unlocks = max(
            0,
            int(lm_response.get("expected_unlocks", selected_pair.unlock_score)),
        )
        tip = f"Рекомендовано: відкриє ~{expected_unlocks} плиток"
        tx = min(selected_pair.first_coords[0], selected_pair.second_coords[0])
        ty = max(
            15,
            min(
                selected_pair.first_coords[1],
                selected_pair.second_coords[1],
            )
            - 18,
        )
        font_path = os.path.join(
            os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "segoeui.ttf"
        )
        try:
            font = ImageFont.truetype(font_path, 12)
        except OSError:
            font = ImageFont.load_default()
        draw.text((tx, ty), tip, fill=(255, 213, 79, 255), font=font)

    rgba = np.asarray(img, dtype=np.uint8)
    bgra = rgba[:, :, [2, 1, 0, 3]].copy()
    return premultiply_bgra(bgra)


# --- Win32: реєстрація класу та вікно ------------------------------------------

_WS_EX_LAYERED = 0x00080000
_WS_EX_TOPMOST = 0x00000008
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_NOACTIVATE = 0x08000000
_WS_POPUP = 0x80000000
_WS_VISIBLE = 0x10000000
_ULW_ALPHA = 0x00000002
_AC_SRC_OVER = 0x00
_AC_SRC_ALPHA = 0x01

_NULL_BRUSH = 5


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class SIZE(ctypes.Structure):
    _fields_ = [("cx", wintypes.LONG), ("cy", wintypes.LONG)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_ubyte),
        ("BlendFlags", ctypes.c_ubyte),
        ("SourceConstantAlpha", ctypes.c_ubyte),
        ("AlphaFormat", ctypes.c_ubyte),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 1),
    ]


_ERROR_CLASS_ALREADY_EXISTS = 1410

_OVERLAY_CLASS_NAME = "MahjongDuelsLayeredOverlayV2"
_overlay_class_registered = False


def _def_window_proc(hwnd, msg, wparam, lparam):
    return ctypes.windll.user32.DefWindowProcW(
        hwnd, msg, wparam, lparam
    )


_WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long,
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)
_wndproc_ref = _WNDPROC(_def_window_proc)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", _WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HANDLE),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HANDLE),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


def _ensure_overlay_window_class() -> None:
    global _overlay_class_registered
    if _overlay_class_registered:
        return

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    gdi32 = ctypes.windll.gdi32

    wc = WNDCLASSW()
    wc.style = 0
    wc.lpfnWndProc = _wndproc_ref
    wc.cbClsExtra = 0
    wc.cbWndExtra = 0
    wc.hInstance = kernel32.GetModuleHandleW(None)
    wc.hIcon = None
    wc.hCursor = user32.LoadCursorW(None, 32512)  # IDC_ARROW
    wc.hbrBackground = gdi32.GetStockObject(_NULL_BRUSH)
    wc.lpszMenuName = None
    wc.lpszClassName = _OVERLAY_CLASS_NAME

    if not user32.RegisterClassW(ctypes.byref(wc)):
        err = kernel32.GetLastError()
        if err != _ERROR_CLASS_ALREADY_EXISTS:
            raise OSError(f"RegisterClassW не вдався, код {err}")
    _overlay_class_registered = True


class Win32LayeredOverlay:
    """Одношарове вікно поверх усіх; кліки там, де alpha=0 на бітмапі."""

    def __init__(self) -> None:
        self._hwnd: wintypes.HWND | None = None

    @property
    def hwnd(self) -> int | None:
        return int(self._hwnd) if self._hwnd else None

    def destroy(self) -> None:
        if self._hwnd:
            ctypes.windll.user32.DestroyWindow(self._hwnd)
            self._hwnd = None

    def show_bitmap(
        self,
        screen_left: int,
        screen_top: int,
        width: int,
        height: int,
        bgra_premul: np.ndarray,
    ) -> None:
        """
        Створює або пересоздає вікно й показує бітмап (premultiplied BGRA, HxWx4).
        """
        self.destroy()
        if width <= 0 or height <= 0:
            return
        if bgra_premul.shape[:2] != (height, width):
            raise ValueError("Розмір бітмапу не збігається з width/height")

        _ensure_overlay_window_class()

        user32 = ctypes.windll.user32
        ex = (
            _WS_EX_LAYERED
            | _WS_EX_TOPMOST
            | _WS_EX_TOOLWINDOW
            | _WS_EX_NOACTIVATE
        )
        hwnd = user32.CreateWindowExW(
            ex,
            _OVERLAY_CLASS_NAME,
            "",
            _WS_POPUP | _WS_VISIBLE,
            int(screen_left),
            int(screen_top),
            int(width),
            int(height),
            None,
            None,
            ctypes.windll.kernel32.GetModuleHandleW(None),
            None,
        )
        if not hwnd:
            raise OSError("CreateWindowExW для оверлею не вдався")
        self._hwnd = hwnd

        self._update_layered(hwnd, screen_left, screen_top, width, height, bgra_premul)

    def _update_layered(
        self,
        hwnd: wintypes.HWND,
        screen_left: int,
        screen_top: int,
        width: int,
        height: int,
        bgra_premul: np.ndarray,
    ) -> None:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        screen_dc = user32.GetDC(None)
        if not screen_dc:
            raise OSError("GetDC не вдався")
        mem_dc = gdi32.CreateCompatibleDC(screen_dc)
        if not mem_dc:
            user32.ReleaseDC(None, screen_dc)
            raise OSError("CreateCompatibleDC не вдався")

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0

        bits = ctypes.c_void_p()
        hbitmap = gdi32.CreateDIBSection(
            screen_dc,
            ctypes.byref(bmi),
            0,
            ctypes.byref(bits),
            None,
            0,
        )
        if not hbitmap or not bits.value:
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(None, screen_dc)
            raise OSError("CreateDIBSection не вдався")

        row_bytes = width * 4
        total = row_bytes * height
        src = bgra_premul.astype(np.uint8, copy=False)
        if not src.flags["C_CONTIGUOUS"]:
            src = np.ascontiguousarray(src)
        ctypes.memmove(bits, src.ctypes.data, total)

        old_obj = gdi32.SelectObject(mem_dc, hbitmap)

        pt_dst = POINT(screen_left, screen_top)
        sz = SIZE(width, height)
        pt_src = POINT(0, 0)
        blend = BLENDFUNCTION(
            _AC_SRC_OVER, 0, 255, _AC_SRC_ALPHA
        )

        # hdcDst = NULL — типовий варіант для ULW_ALPHA (екранний DC підставляє система).
        ok = user32.UpdateLayeredWindow(
            hwnd,
            None,
            ctypes.byref(pt_dst),
            ctypes.byref(sz),
            mem_dc,
            ctypes.byref(pt_src),
            0,
            ctypes.byref(blend),
            _ULW_ALPHA,
        )

        gdi32.SelectObject(mem_dc, old_obj)
        gdi32.DeleteObject(hbitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(None, screen_dc)

        if not ok:
            raise OSError("UpdateLayeredWindow не вдався")


def is_supported() -> bool:
    return sys.platform == "win32"
