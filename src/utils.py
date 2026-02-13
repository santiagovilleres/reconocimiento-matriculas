import cv2

def pre_procesar(recorte):
    """
    Aplica filtros y mantiene la compatibilidad de canales para el modelo.
    """
    if recorte is None or recorte.size == 0:
        return recorte
    
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(gris, (3, 3), 0)
    recorte_final = cv2.cvtColor(suavizado, cv2.COLOR_GRAY2BGR)
    
    return recorte_final