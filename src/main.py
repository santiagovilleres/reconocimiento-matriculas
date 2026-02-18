from detect import Detector
import tkinter
from tkinter import filedialog

RUTA_MODELO = "models/weights/best.pt"
NOMBRE_MODELO_OCR = "cct-xs-v1-global-model"

IMAGEN = "1"
CARPETA = "2"
WEBCAM = "3"
SALIR = "4"

def main():

    detector = Detector(ruta_modelo=RUTA_MODELO, nombre_modelo_ocr=NOMBRE_MODELO_OCR)

    while True:

        print("Seleccione una opción:")
        print("1 : Procesar imagen")
        print("2 : Procesar carpeta")
        print("3 : Webcam")
        print("4 : Salir")

        opcion = input("Opción: ")
        
        if opcion == SALIR:
            break

        root = tkinter.Tk()

        root.withdraw()

        if opcion == IMAGEN:
            ruta = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Imagenes", "*.jpg *.jpeg *.png")])

        elif opcion == CARPETA:
            ruta = filedialog.askdirectory(title="Seleccionar carpeta")

        elif opcion == WEBCAM:
            detector.procesar_webcam()
            return
        
        else:
            print("Opción inválida")
            return

        if ruta:
            detector.procesar_imagen(ruta)

        else:
            print("No se seleccionó nada")

if __name__ == "__main__":
    main()