import cv2
import numpy as np
from pathlib import Path

import cv2
import numpy as np
from pathlib import Path

def process_image(image_path: Path) -> Path:
    """
    Detecta el marco amarillo en la imagen y recorta la región interior sin incluir el marco.
    """
    # Leer la imagen
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("No se pudo leer la imagen.")

    # Convertir a espacio de color HSV para identificar el amarillo
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Aplicar detección de bordes
    edges = cv2.Canny(mask, 50, 150)

    # Encontrar líneas con la Transformada de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is None:
        raise ValueError("No se encontraron bordes del marco.")

    # Dibujar las líneas detectadas para verificar (opcional)
    line_image = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Encontrar los puntos extremos del marco (intersección de líneas)
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.extend([(x1, y1), (x2, y2)])

    # Calcular el bounding box interior basado en los puntos
    points = np.array(points)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    # Recortar el interior del marco
    cropped = image[y_min:y_max, x_min:x_max]

    # Guardar la imagen de debugging
    debug_path = image_path.parent / f"debug_{image_path.name}"
    cv2.imwrite(str(debug_path), line_image)

    # Guardar la imagen procesada
    processed_path = image_path.parent / f"processed_{image_path.name}"
    cv2.imwrite(str(processed_path), cropped)
    
    return processed_path


def detect_frame(image_path: Path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("La imagen no se pudo cargar.")

    # Convertir a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Máscara para amarillo con rango amplio
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Aplicar gradiente Sobel para detectar bordes
    grad_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = np.uint8(grad)

    # Detectar contornos
    contours, _ = cv2.findContours(grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por aproximación a rectángulos
    rectangles = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # Rectángulo detectado
            rectangles.append(approx)

    # Seleccionar el rectángulo más grande
    if rectangles:
        largest_rect = max(rectangles, key=cv2.contourArea)

        # Dibujar el rectángulo en la imagen original (debug)
        debug_image = image.copy()
        debug_path = image_path.parent / f"debug_{image_path.name}"
        cv2.drawContours(debug_image, [largest_rect], -1, (0, 255, 0), 2)
        cv2.imwrite(f"{debug_path}/detected_rectangle.jpg", debug_image)

        # Extraer la región interior del marco
        pts = largest_rect.reshape(4, 2)
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        extracted = image[y:y+h, x:x+w]

        cv2.imwrite(f"{debug_path}/extracted_region.jpg", extracted)

        return f"Marco detectado y región extraída guardada en {debug_path}"
    else:
        return "No se detectaron rectángulos. Ajusta los parámetros o revisa la calidad de la imagen."