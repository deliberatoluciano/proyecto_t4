import cv2
import numpy as np

# Cargo imagen del motor
image = cv2.imread('motor_3.png', cv2.IMREAD_COLOR)

# Convierto la escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplico el desenfoque gaussiano para reducir el ruido
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Detector de bordes Canny
edges = cv2.Canny(blurred, 50, 150)

# Transformada de Hough para rectas
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

# Dibujo las líneas en la imagen original
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Muestro la imagen con las líneas detectadas
cv2.imshow('Detected Lines', image)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Si se detectaron líneas, imprimir sus puntos de inicio y fin
if lines is not None:
    for line in lines:
        print(f"Line: {line[0]}")