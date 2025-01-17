import cv2
import numpy as np
import os

def process_image(input_path):
    # Crear directorio de salida
    output_dir = "app/static"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Leer la imagen
    image = cv2.imread(input_path)
    original_debug_path = os.path.join(output_dir, "original_image.png")
    cv2.imwrite(original_debug_path, image)  # Guardar la imagen original

    # Convertir la imagen a espacio de color LAB para filtrar por luminosidad
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Separar los canales L (luminosidad), A (verde-rojo) y B (azul-amarillo)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Establecer un umbral para la luminosidad (L), para detectar las zonas más claras
    _, light_mask = cv2.threshold(l_channel, 250, 255, cv2.THRESH_BINARY)  # Ajusta 180 para mayor o menor claridad

    # Aplicar la máscara para obtener solo las áreas más claras
    light_part = cv2.bitwise_and(image, image, mask=light_mask)

    # Guardar la imagen filtrada por luminosidad
    light_filtered_debug_path = os.path.join(output_dir, "light_filtered_image.png")
    cv2.imwrite(light_filtered_debug_path, light_part)

    # Aplicar un filtro para reducir el ruido
    blurred_image = cv2.GaussianBlur(light_part, (15, 15), 0)  # Usando Gaussian Blur

    # Guardar la imagen después de aplicar el blur
    blurred_image_debug_path = os.path.join(output_dir, "blurred_image.png")
    cv2.imwrite(blurred_image_debug_path, blurred_image)

    # Convertir a escala de grises para detectar bordes
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Umbral para detectar bordes después del suavizado
    _, edges = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Guardar la imagen después de la detección de bordes
    edges_debug_path = os.path.join(output_dir, "edges_image.png")
    cv2.imwrite(edges_debug_path, edges)

    # Encontrar las líneas en la imagen con la Transformada de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Filtrar las líneas verticales y horizontales
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calcular el ángulo de la línea
        delta_x = x2 - x1
        delta_y = y2 - y1
        angle = np.arctan2(delta_y, delta_x) * 180 / np.pi  # Ángulo en grados

        # Si el ángulo es cercano a 0 o 180 grados, es una línea vertical
        if abs(angle) < 1 or abs(angle - 180) < 1:
            vertical_lines.append((x1, y1, x2, y2))
        # Si el ángulo es cercano a 90 o 270 grados, es una línea horizontal
        elif abs(angle - 90) < 5 or abs(angle - 270) < 5:
            horizontal_lines.append((x1, y1, x2, y2))

    # Dibujar solo las líneas verticales y horizontales en la imagen filtrada
    filtered_lines_image = image.copy()
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        cv2.line(filtered_lines_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Azul para verticales
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        cv2.line(filtered_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Rojo para horizontales

    filtered_lines_debug_path = os.path.join(output_dir, "filtered_lines.png")
    cv2.imwrite(filtered_lines_debug_path, filtered_lines_image)

    # Encontrar el rectángulo que rodea las líneas (esto será el borde del cuadrado)
    # Extraer las coordenadas mínimas y máximas de las líneas
    all_x = []
    all_y = []
    for line in vertical_lines + horizontal_lines:
        x1, y1, x2, y2 = line
        all_x.extend([x1, x2])
        all_y.extend([y1, y2])

    # Calcular el rectángulo de interés
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Desplazar el rectángulo hacia el interior para enfocarse en la parte dentro del cuadrado
    offset = 60  # Ajusta este valor para mover más o menos hacia el interior
    min_x, max_x = min_x + offset, max_x - offset
    min_y, max_y = min_y + offset, max_y - offset

    # Recortar la imagen para centrarse solo en el cuadrado
    cropped_image = image[min_y:max_y, min_x:max_x]

    # Guardar la imagen recortada
    cropped_image_path = os.path.join(output_dir, "cropped_image.png")
    cv2.imwrite(cropped_image_path, cropped_image)

    # Focus on the center by adjusting the cropping area
    center_x, center_y = cropped_image.shape[1] // 2, cropped_image.shape[0] // 2
    crop_size = 850# Define the size of the center focus area

    # Recortar alrededor del centro (esto debería enfocarse en el cuadrado)
    min_x, max_x = center_x - crop_size, center_x + crop_size
    min_y, max_y = center_y - crop_size, center_y + crop_size
    center_cropped_image = cropped_image[min_y:max_y, min_x:max_x]

    # Guardar la imagen centrada
    center_cropped_image_path = os.path.join(output_dir, "center_cropped_image.png")
    cv2.imwrite(center_cropped_image_path, center_cropped_image)

    return "Proceso completo. Las imágenes de depuración están disponibles en la carpeta de salida."
