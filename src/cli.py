from detect import Detector
from ocr import FastPlateOCR
from pipeline import run_webcam, run_image, run_folder

import tkinter as tk
from tkinter import filedialog


def seleccionar_imagen():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imagenes", "*.jpg *.jpeg *.png")]
    )

    root.destroy()
    return file_path


def seleccionar_carpeta():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder_path = filedialog.askdirectory(
        title="Seleccionar carpeta"
    )

    root.destroy()
    return folder_path


def main():
    detector = Detector("models/weights/best.pt")
    ocr = FastPlateOCR()

    while True:
        print("\n===== ALPR - Reconocimiento de Patentes =====")
        print("1 - Modo Webcam")
        print("2 - Procesar una imagen")
        print("3 - Procesar una carpeta")
        print("0 - Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            run_webcam(detector, ocr)

        elif opcion == "2":
            ruta = seleccionar_imagen()
            if ruta:
                run_image(ruta, detector, ocr)

        elif opcion == "3":
            ruta = seleccionar_carpeta()
            if ruta:
                run_folder(ruta, detector, ocr)

        elif opcion == "0":
            break

        else:
            print("Opción inválida.")


if __name__ == "__main__":
    main()
