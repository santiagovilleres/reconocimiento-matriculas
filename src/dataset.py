import os

CARPETA_IMAGENES = "data/images"
CARPETA_LABELS = "data/labels"

def renombrar():
    imagenes = sorted([f for f in os.listdir(CARPETA_IMAGENES) if f.endswith(".jpg")])
    contador = 1

    for img in imagenes:
        nombre_base = os.path.splitext(img)[0]
        label = nombre_base + ".txt"

        ruta_imagen = os.path.join(CARPETA_IMAGENES, img)
        ruta_label = os.path.join(CARPETA_LABELS, label)

        if os.path.exists(ruta_label):
            nuevo_nombre = f"{contador:03d}"

            os.rename(ruta_imagen, os.path.join(CARPETA_IMAGENES, nuevo_nombre + ".jpg"))
            os.rename(ruta_label, os.path.join(CARPETA_LABELS, nuevo_nombre + ".txt"))

            contador += 1

if __name__ == "__main__":
    renombrar()
