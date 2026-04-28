from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from vision import TileMatch, VisionEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Тест Vision: порівнює кадр гри з шаблонами з assets/tiles "
            "і друкує знайдені плитки з координатами."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        default="game_frame.jpg",
        help="Шлях до тестового кадру гри (за замовчуванням: game_frame.jpg).",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="assets/tiles",
        help="Папка з шаблонами плиток.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.88,
        help="Поріг збігу для cv2.matchTemplate.",
    )
    parser.add_argument(
        "--template-scale",
        type=float,
        default=1.0,
        help=(
            "Масштаб усіх шаблонів відносно файлів (напр. 0.35, якщо кадр з меншими плитками). "
            "Див. find_template_scale.py для підбору."
        ),
    )
    parser.add_argument(
        "--save-debug",
        type=str,
        default="vision_debug.jpg",
        help="Куди зберегти кадр з рамками (для перевірки).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Максимум рядків із детальними збігами в консолі.",
    )
    return parser


def print_match(match: TileMatch) -> None:
    print(
        f"{match.tile_type:20s} "
        f"conf={match.confidence:.3f} "
        f"x={match.x:4d} y={match.y:4d} w={match.w:3d} h={match.h:3d}"
    )


def main() -> None:
    args = build_parser().parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Не знайдено файл кадру: {image_path.resolve()}\n"
            "Зробіть скріншот гри й покладіть його в корінь проєкту, "
            "або передайте інший шлях через --image."
        )

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError(f"Не вдалося прочитати зображення: {image_path.resolve()}")

    engine = VisionEngine(
        templates_dir=args.templates,
        threshold=args.threshold,
        template_scale=args.template_scale,
    )
    matches, grouped = engine.find_templates(frame)

    print("=" * 80)
    print("РЕЗУЛЬТАТ ТЕСТУ VISION")
    print(f"Файл кадру: {image_path.resolve()}")
    print(f"Папка шаблонів: {Path(args.templates).resolve()}")
    print(f"Поріг threshold: {args.threshold}")
    print(f"Масштаб шаблонів template_scale: {args.template_scale}")
    print(f"Усього збігів: {len(matches)}")
    print(f"Унікальних типів: {len(grouped)}")
    print("=" * 80)

    # Спочатку виводимо статистику за типами.
    for tile_type in sorted(grouped.keys()):
        print(f"{tile_type:20s} -> {len(grouped[tile_type])} шт.")

    # Далі — детальні координати.
    if matches:
        print("\nДетальні збіги (тип, confidence, координати):")
        ordered = sorted(matches, key=lambda m: (m.tile_type, -m.confidence))
        for idx, match in enumerate(ordered):
            if idx >= args.limit:
                print(f"... обрізано вивід після {args.limit} рядків")
                break
            print_match(match)

    debug_img = VisionEngine.draw_matches(frame, matches)
    cv2.imwrite(str(args.save_debug), debug_img)
    print(f"\nЗбережено debug-зображення: {Path(args.save_debug).resolve()}")


if __name__ == "__main__":
    main()
