import cv2
import numpy as np
import os
from PIL import Image

# КОНФІГУРАЦІЯ
INPUT_IMAGE = 'Screenshot 2026-04-25 094829.png'  # Шлях до вашого скріншоту
OUTPUT_DIR = 'assets/tiles_templates'  # Папка для збереження плиток

# Приблизні розміри плитки (width, height). Можливо, доведеться підкоригувати.
TILE_WIDTH_TARGET = 60
TILE_HEIGHT_TARGET = 80

def create_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Створено папку: {directory}")

def load_source_image(path):
    # Використовуємо OpenCV для завантаження, але зберігши колір (BGRA)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Помилка: Не вдалося завантажити {path}. Перевірте шлях.")
        exit()
    print(f"Зображення завантажено. Розмір: {image.shape[1]}x{image.shape[0]}")
    return image

def preprocess_image(image):
    # Перетворюємо в сірий
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Інвертуємо кольори, щоб фон був чорним, а плитки — білими (якщо фон світлий)
    # Це допомагає знайти контури.
    gray_inv = cv2.bitwise_not(gray)

    # Порогова обробка (thresholding) для отримання бінарного зображення (чорно-білого)
    _, binary = cv2.threshold(gray_inv, 100, 255, cv2.THRESH_BINARY)
    
    # Використовуємо морфологічні операції для очищення шуму та об'єднання частин плиток
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_morphed = cv2.morphologyEx(binary_morphed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Для діагностики можна розкоментувати:
    # cv2.imshow("Morphed Binary", binary_morphed)
    # cv2.waitKey(0)
    
    return binary_morphed

def extract_and_save_tiles(source_image, binary_mask, output_dir):
    # Знаходимо контури
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фільтруємо контури, які занадто малі або занадто великі для плитки
    # Оскільки на скріншоті плитки акуратно розкладені, це має спрацювати добре.
    # Ми шукаємо контури, чия площа приблизно відповідає площі плитки.
    target_area = TILE_WIDTH_TARGET * TILE_HEIGHT_TARGET
    min_area = target_area * 0.5
    max_area = target_area * 2.0
    
    tile_count = 0
    tile_images = []

    # Створюємо копію для малювання (діагностика)
    diagnostics_image = source_image.copy()

    # Проходимо по всіх контурах, але у зворотному порядку, щоб йти зверху вниз, зліва направо (OpenCV йде знизу вгору)
    sorted_contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]))

    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w)/h
        
        # Перевіряємо умови: площа та пропорції
        # Маджонг плитки зазвичай вищі, ніж широкі (aspect ratio < 1)
        if min_area < area < max_area and 0.5 < aspect_ratio < 0.9:
            # Вирізаємо плитку з оригінального кольорового зображення
            # (додаємо невеликий відступ, щоб не обрізати краї)
            padding = 2
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(source_image.shape[1], x + w + padding), min(source_image.shape[0], y + h + padding)
            
            roi = source_image[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Змінюємо розмір до еталонного
                resized_tile = cv2.resize(roi, (TILE_WIDTH_TARGET, TILE_HEIGHT_TARGET))
                
                # Малюємо прямокутник для діагностики
                cv2.rectangle(diagnostics_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(diagnostics_image, str(tile_count), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                tile_images.append(resized_tile)
                tile_count += 1

    # cv2.imshow("Detected Tiles", diagnostics_image)
    # cv2.waitKey(0)

    # Зберігаємо знайдені плитки
    for i, tile_img in enumerate(tile_images):
        tile_name = f"tile_{i:03d}.png"
        tile_path = os.path.join(output_dir, tile_name)
        # OpenCV зберігає як BGR, тому для Pillow (якщо потрібно) треба конвертувати,
        # але ми зберігаємо через OpenCV для простоти.
        cv2.imwrite(tile_path, tile_img)
        print(f"Збережено: {tile_path}")

    print(f"Всього знайдено та збережено плиток: {tile_count}")

# ЗАПУСК
if __name__ == "__main__":
    create_output_dir(OUTPUT_DIR)
    img_bgr = load_source_image(INPUT_IMAGE)
    processed_mask = preprocess_image(img_bgr)
    extract_and_save_tiles(img_bgr, processed_mask, OUTPUT_DIR)