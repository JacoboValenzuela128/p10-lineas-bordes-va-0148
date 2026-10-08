import cv2
import numpy as np
import os

# Obtener la ruta del directorio actual ('codigo')
DIR_CODIGO = os.path.dirname(os.path.abspath(__file__))

# Definir rutas absolutas
RUTA_IMAGEN = os.path.normpath(os.path.join(DIR_CODIGO, "..", "imagenes", "lineas_esquinas.jpg"))
RUTA_SALIDA = os.path.normpath(os.path.join(DIR_CODIGO, "..", "resultados", "ejemplo1_lineas.jpg"))

# Cargar la imagen
imagen = cv2.imread(RUTA_IMAGEN)

if imagen is None:
    print("Error: No se pudo cargar la imagen desde:", RUTA_IMAGEN)
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar detección de bordes Canny
bordes = cv2.Canny(gris, 50, 150, apertureSize=3)

# Detección de líneas mediante la Transformada Probabilística de Hough
lineas = cv2.HoughLinesP(
    bordes,
    rho=1,
    theta=np.pi / 180,
    threshold=100,
    minLineLength=50,
    maxLineGap=10
)

# Crear una copia de la imagen para dibujar los resultados
resultado = imagen.copy()

# Dibujar las líneas detectadas en color verde
cantidad_lineas = 0
if lineas is not None:
    cantidad_lineas = len(lineas)
    for linea in lineas:
        # Extraer coordenadas de forma segura desempacando el arreglo aplanado
        x1, y1, x2, y2 = linea.ravel()
        cv2.line(resultado, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Mostrar resultados
cv2.imshow("Imagen original", imagen)
cv2.imshow("Lineas detectadas", resultado)

# Guardar resultado
os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
cv2.imwrite(RUTA_SALIDA, resultado)

print("Deteccion de lineas terminada.")
print(f"Cantidad de lineas detectadas: {cantidad_lineas}")
print("Resultado guardado en:", RUTA_SALIDA)

cv2.waitKey(0)
cv2.destroyAllWindows()