import cv2
import numpy as np

class Panel():
    def __init__(self, x1, y1, x2, y2, score=0.0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.score = score

    def _obtener_area(self):
        ancho = max(0, self.x2 - self.x1)
        alto = max(0, self.y2 - self.y1)

        return ancho * alto

    def _calcular_iou(self, otro_panel):
        """
        Intersección sobre Unión entre este panel y otro. Eliminamos detecciones repetidas
        """

        # coords de intersección
        inter_x1 = max(self.x1, otro_panel.x1)
        inter_y1 = max(self.y1, otro_panel.y1)
        inter_x2 = min(self.x2, otro_panel.x2)
        inter_y2 = min(self.y2, otro_panel.y2)

        # area de intersección
        inter_ancho = max(0, inter_x2 - inter_x1)
        inter_alto = max(0, inter_y2 - inter_y1)
        area_interseccion = inter_ancho * inter_alto

        # area de unión
        area_union = self._obtener_area() + otro_panel._obtener_area() - area_interseccion

        # IoU = area_overlap / area_union
        return area_interseccion / area_union
