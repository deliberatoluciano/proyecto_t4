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

# Transformada de Hough para círculos
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=0, maxRadius=0)

# Verifico si se detectó un círculo
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Dibujo el círculo externo
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Dibujo el centro del círculo
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# Muestro la imagen con los círculos detectados
cv2.imshow('Detected Circles', image)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Si se detectó algún circulo, muestro sus coordenadas y radio
if circles is not None:
    for circle in circles[0, :]:
        print(f"Circle center (X, Y): ({circle[0]}, {circle[1]})")
        print(f"Circle radius: {circle[2]}")