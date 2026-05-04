import argparse
import os
import cv2

from resultados import GestorResultados
from detectores import DetectorBasico, DetectorAvanzado

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, nargs="?", default="basico")
    parser.add_argument('--train_path', default="")
    parser.add_argument('--test_path', default="")
    
    args = parser.parse_args()

    gestor = GestorResultados()

    if args.detector.lower() == "avanzado":
        detector = DetectorAvanzado()
    else:
        detector = DetectorBasico()

    if os.path.exists(args.test_path):
        archivos = os.listdir(args.test_path)
        imagenes_png = [f for f in archivos if f.lower().endswith('.png')]
        
        for nombre_fichero in sorted(imagenes_png):
            ruta_img = os.path.join(args.test_path, nombre_fichero)
            imagen = cv2.imread(ruta_img)
            
            if imagen is not None:
                paneles_detectados = detector.detectar(imagen)
                gestor.guardar_detecciones(nombre_fichero, imagen, paneles_detectados)


