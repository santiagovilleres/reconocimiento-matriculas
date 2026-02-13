from detect import DetectorMatrículas
from ocr import LectorMatrícula
from pipeline import ejecutar_webcam, ejecutar_imagen, ejecutar_carpeta

import tkinter as tk
from tkinter import filedialog


def seleccionar_ruta(es_carpeta=False):
    ventana = tk.Tk()
    ventana.withdraw()
    ventana.attributes("-topmost", True)

    if es_carpeta:
        ruta = filedialog.askdirectory(title="Seleccionar carpeta")
    else:
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imagenes", "*.jpg *.jpeg *.png")]
        )

    ventana.destroy()
    return ruta


def main():
    detector = DetectorMatrículas("models/weights/best.pt")
    lector = LectorMatrícula()

    while True:
        print("\n===== ALPR - Reconocimiento de Patentes =====")
        print("1 - Modo Webcam")
        print("2 - Procesar una imagen")
        print("3 - Procesar una carpeta")
        print("0 - Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            ejecutar_webcam(detector, lector)

        elif opcion == "2":
            ruta = seleccionar_ruta(es_carpeta=False)
            if ruta:
                ejecutar_imagen(ruta, detector, lector)

        elif opcion == "3":
            ruta = seleccionar_ruta(es_carpeta=True)
            if ruta:
                ejecutar_carpeta(ruta, detector, lector)

        elif opcion == "0":
            break

        else:
            print("Opción inválida.")


if __name__ == "__main__":
    main()
