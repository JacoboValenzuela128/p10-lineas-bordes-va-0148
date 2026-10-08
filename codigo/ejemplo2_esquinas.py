import cv2
import numpy as np
import os

# Obtener la ruta del directorio actual donde está este archivo .py (carpeta 'codigo')
DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))

# Construir rutas absolutas a partir del script
RUTA_IMAGEN = os.path.join(DIR_ACTUAL, "..", "imagenes", "lineas_esquinas.jpg")
RUTA_SALIDA = os.path.join(DIR_ACTUAL, "..", "resultados", "ejemplo2_esquinas.jpg")

# Cargar imagen
imagen = cv2.imread(RUTA_IMAGEN)

# Verificar que la imagen exista
if imagen is None:
    print(f"Error: no se pudo cargar la imagen desde {RUTA_IMAGEN}")
    print("Verifica que la imagen exista en la carpeta 'imagenes' y se llame 'lineas_esquinas.jpg'.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a tipo float32
gris_float = np.float32(gris)

# Detectar esquinas mediante Harris
esquinas = cv2.cornerHarris(
    gris_float,
    2,
    3,
    0.04
)

# Dilatar para hacer visibles las esquinas
esquinas = cv2.dilate(
    esquinas,
    None
)

# Crear copia
resultado = imagen.copy()

# Umbral para identificar esquinas
umbral = 0.01 * esquinas.max()

# Marcar esquinas en rojo
resultado[esquinas > umbral] = [0, 0, 255]

# Mostrar resultados
cv2.imshow("Imagen original", imagen)
cv2.imshow("Esquinas detectadas", resultado)

# Guardar resultado
cv2.imwrite(RUTA_SALIDA, resultado)

# Contar esquinas aproximadas
cantidad_esquinas = np.sum(esquinas > umbral)

print("Deteccion de esquinas terminada.")
print("Cantidad aproximada de puntos detectados:", cantidad_esquinas)
print("Resultado guardado en:", RUTA_SALIDA)

# Esperar una tecla
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()