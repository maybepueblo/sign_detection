import os
import cv2

class GestorResultados:
    def __init__(self, ruta_salida_imgs="resultado_imgs", archivo_txt="resultado.txt"):
        self.ruta_salida = ruta_salida_imgs
        self.archivo_txt = archivo_txt
        self._preparar_entorno()

    def _preparar_entorno(self):
        """Crea el directorio 'resultado_imgs' si no existe y limpia el txt[cite: 1]."""
        if not os.path.exists(self.ruta_salida):
            os.makedirs(self.ruta_salida)
        # Limpiar o crear el fichero resultado.txt al inicio de la ejecución
        open(self.archivo_txt, 'w').close()

    def guardar_detecciones(self, nombre_fichero, imagen_original, lista_paneles):
        # Hacer una copia para no modificar la imagen original en memoria
        img_dibujada = imagen_original.copy()

        for panel in lista_paneles:
            # 1. Dibujar el rectángulo en ROJO (BGR: 0, 0, 255)
            cv2.rectangle(img_dibujada, (panel.x1, panel.y1), (panel.x2, panel.y2), (0, 0, 255), 2)

            # 2. Escribir el score en AMARILLO (BGR: 0, 255, 255)
            texto_score = f"{panel.score:.2f}"
            
            # Calcular posición del texto (encima del rectángulo, o debajo si choca con el borde superior)
            pos_y = panel.y1 - 10 if panel.y1 > 20 else panel.y1 + 20
            
            cv2.putText(img_dibujada, texto_score, (panel.x1, pos_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 3. Guardar la imagen en la carpeta 'resultado_imgs'
        ruta_guardado = os.path.join(self.ruta_salida, nombre_fichero)
        cv2.imwrite(ruta_guardado, img_dibujada)

        # 4. Registrar las detecciones en el txt
        self._escribir_txt(nombre_fichero, lista_paneles)

    def _escribir_txt(self, nombre_fichero, lista_paneles):
        """
        Escribe en el fichero txt siguiendo el formato estricto:
        <nombre_fichero>;<x1>;<y1>;<x2>;<y2>;1;<score>[cite: 1].
        """
        with open(self.archivo_txt, 'a') as f:
            for p in lista_paneles:
                # El identificador de clase siempre es 1[cite: 1]
                linea = f"{nombre_fichero};{p.x1};{p.y1};{p.x2};{p.y2};1;{p.score:.3f}\n"
                f.write(linea)