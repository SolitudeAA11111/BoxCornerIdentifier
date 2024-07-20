import cv2
import numpy as np
import json
from math import sqrt

def execute(image_path, output_filename="app/src/output/output.jpg"):
    """Запуск полного процесса обработки изображения."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")

    min_area = 1800
    max_area = 40000
    window_title = 'Original Image'
    detected_boxes = []

    # Удаление теней с изображения.
    planes = cv2.split(image)
    result_planes = []
    norm_planes = []
    for plane in planes:
        dilated_plane = cv2.dilate(plane, np.ones((5, 5), np.uint8))
        blurred_bg = cv2.medianBlur(dilated_plane, 19)
        diff_img = 255 - cv2.absdiff(plane, blurred_bg)
        normalized_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        result_planes.append(diff_img)
        norm_planes.append(normalized_img)

    image_without_shadows = cv2.merge(result_planes)

    # Применение детекции краев и морфологических операций.
    edges = cv2.Canny(image_without_shadows, 55, 150)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

    # Поиск контуров на изображении.
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Выделение контуров на изображении в зависимости от площади.
    contours = sorted(contours, key=cv2.contourArea)[:-1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            if _is_valid_box(contour, box):
                cv2.drawContours(image, [box.astype(int)], -1, (0, 0, 255), 2)
                detected_boxes.append(box)


    # Сохранение обработанного изображения в файл.
    # Отображение изображения в новом окне.
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_title)
    cv2.imwrite(output_filename, image)

    # Создание JSON-выходных данных с обнаруженными прямоугольниками.
    boxes_data = {
        idx: {"x1": float(box[0][0]), "y1": float(box[0][1]),
              "x2": float(box[1][0]), "y2": float(box[1][1]),
              "x3": float(box[2][0]), "y3": float(box[2][1]),
              "x4": float(box[3][0]), "y4": float(box[3][1])}
        for idx, box in enumerate(detected_boxes)
    }
    with open("app/src/output/output.json", "w") as json_file:
        json.dump(boxes_data, json_file)
    return boxes_data

def _is_valid_box(contour, box):
    """Проверка валидности контурного прямоугольника по расстоянию точек."""
    tolerance = 60
    x, y, w, h = cv2.boundingRect(contour)
    bounding_points = [(x, y+h), (x, y), (x+w, y), (x+w, y+h)]
    return all(_euclidean_distance(p, bp) < tolerance for p, bp in zip(bounding_points, box))

def _euclidean_distance(point1, point2):
    """Расчет Евклидова расстояния между двумя точками."""
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
