import cv2
import numpy as np
from panel import Panel

class SignDetector():
    def __init__(self):
        # parámetros configuración
        self.tamano_fijo = (80, 40)
        self.umbral_color = 0.3
        
        # parámetros centralizados para detección
        self.confidence_threshold = 0.60
        self.delta = 12                     # MSER más sensible para niebla
        self.min_area = 140 
        self.max_area = 300000 
        self.iou_threshold = 0.22           # Umbral para NMS
        self.aspect_ratio_range = (0.3, 8.25)
        self.margin_expansion = 0.05        # Margen para no cortar bordes
        self.line_size = 0.10             
        self.num_sides = 4 

        # clahe para luz dinámica
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def _detectar(self, imagen):   
        """
        Metodo principal. Debe recibir una imagen y devolver una lista de objetos Panel.
        Este método será sobreescrito por las clases hijas
        """ 
        raise NotImplementedError("Método implementado por hijos")

class DetectorBasico(SignDetector):
    def __init__(self):
        super().__init__()
        # inicializar MSER usando los parámetros de la clase base
        self.mser = cv2.MSER_create(
            delta=self.delta, 
            min_area=self.min_area, 
            max_area=self.max_area
        )
        # máscara ideal: np.array de tamaño fijo (40x80) con píxeles a 1
        self.mascara_ideal = np.ones((self.tamano_fijo[1], self.tamano_fijo[0]), dtype=np.uint8)

    def detectar(self, imagen):
        """_summary_

        Args:
            imagen (np.ndarray): _description_

        Returns:
            List[Panel]: _description_
        """
        paneles_detectados = []
        candidatos = self._extraer_candidatos_mser(imagen)

        for candidato in candidatos:
            score = self._validar_color_y_correlacion(imagen, candidato)
            # umbral de confianza
            if score > self.confidence_threshold: 
                paneles_detectados.append(Panel(candidato[0], candidato[1], candidato[2], candidato[3], score))

        return self._eliminar_repetidas(paneles_detectados)

    def _extraer_candidatos_mser(self, imagen):
        """_summary_

        Args:
            imagen (np.ndarray): _description_

        Returns:
            List[Tuple[int, int, int, int]]: _description_
        """
        # 1. preprocesamiento
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_norm = self.clahe.apply(l)

        lab_norm = cv2.merge((l_norm, a, b))
        imagen_mejorada = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

        imagen_gris = cv2.cvtColor(imagen_mejorada, cv2.COLOR_BGR2GRAY)

        # pizco de difuminado para quitarnos letras blancas
        imagen_gris = cv2.GaussianBlur(imagen_gris, (7, 7), 0)

        # 2. detección
        regions, _ = self.mser.detectRegions(imagen_gris)

        candidatos = []
        alto_img, ancho_img = imagen.shape[:2]

        for region in regions:
            x, y, w, h = cv2.boundingRect(region)

            if h == 0 or w < 30 or h < 30 or w > ancho_img * 0.8:
                continue

            # 4. filtrado por relación de aspecto usando el rango centralizado
            aspect_ratio = w / h
            ar_min, ar_max = self.aspect_ratio_range
            
            if ar_min < aspect_ratio < ar_max:
                # expansión de márgenes usando el parámetro margin_expansion
                margen_x = int(w * self.margin_expansion)
                margen_y = int(h * self.margin_expansion)

                x1 = max(0, x - margen_x)
                y1 = max(0, y - margen_y)
                x2 = min(ancho_img, x + w + margen_x)
                y2 = min(alto_img, y + h + margen_y)

                candidatos.append((x1, y1, x2, y2))

        return candidatos

    def _validar_color_y_correlacion(self, imagen, bbox):
        """_summary_

        Args:
            imagen (np.ndarray): _description_
            bbox (Tuple[int, int, int, int]): _description_

        Returns:
            float: _description_
        """
        x1, y1, x2, y2 = bbox
        recorte = imagen[y1:y2, x1:x2]

        if recorte.size == 0:
            return 0.0

        recorte_resized = cv2.resize(recorte, self.tamano_fijo)
        hsv = cv2.cvtColor(recorte_resized, cv2.COLOR_BGR2HSV)

        # rango de azul saturado
        azul_bajo = np.array([100, 120, 50])
        azul_alto = np.array([130, 255, 255])

        mascara_azul = cv2.inRange(hsv, azul_bajo, azul_alto)

        # rellenar letras blancas para crear bloque sólido 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mascara_azul = cv2.morphologyEx(mascara_azul, cv2.MORPH_CLOSE, kernel)

        M = mascara_azul / 255.0

        # correlación con máscara ideal
        correlacion = np.sum(M * self.mascara_ideal)
        total_pixeles = self.tamano_fijo[0] * self.tamano_fijo[1]
        score = correlacion / total_pixeles

        return float(score)

    def _eliminar_repetidas(self, lista_paneles):
        """_summary_

        Args:
            lista_paneles (List[Panel]): _description_
    
        Returns:
            List[Panel]: _description_
        """
        if not lista_paneles:
            return []

        paneles_ordenados = sorted(lista_paneles, key=lambda p: p.score, reverse=True)
        paneles_filtrados = []

        while paneles_ordenados:
            mejor_panel = paneles_ordenados.pop(0)
            paneles_filtrados.append(mejor_panel)

            paneles_restantes = []
            
            for panel in paneles_ordenados:
                iou = mejor_panel._calcular_iou(panel)
                
                # ¿qué porcentaje del panel pequeño está sepultado por el grande?
                iz = max(mejor_panel.x1, panel.x1)
                ar = max(mejor_panel.y1, panel.y1)
                de = min(mejor_panel.x2, panel.x2)
                ab = min(mejor_panel.y2, panel.y2)
                inter = max(0, de - iz + 1) * max(0, ab - ar + 1)
                
                area_panel = (panel.x2 - panel.x1 + 1) * (panel.y2 - panel.y1 + 1)
                iom = inter / float(area_panel + 1e-6) # Intersección / Área de la caja pequeña
            
                # Sobrevive si el IoU es bajo (no se solapan mucho) 
                # Y si el IoM es bajo (no está contenida dentro del cartel grande)
                if iou <= self.iou_threshold and iom <= 0.70:
                    paneles_restantes.append(panel)
            
            paneles_ordenados = paneles_restantes

        return paneles_filtrados

class DetectorAvanzado(SignDetector):
    def __init__(self):
        super().__init__()
        
        # configuración
        self.ancho, self.alto = (80, 40)
        self.margen_x = int(self.ancho * 0.10)
        self.margen_y = int(self.alto * 0.10)
        
        self.zona_interna = np.zeros((self.alto, self.ancho), dtype=bool)
        self.zona_interna[self.margen_y:-self.margen_y, self.margen_x:-self.margen_x] = True
        self.zona_externa = ~self.zona_interna
        
        self.confidence_threshold = 0.50
    
    def _get_mser_adaptive(self, imagen_gris):
        """_summary_ MSER ADAPTATIVO

        Args:
            imagen_gris (np.ndarray): _description_

        Returns:
            cv2.MSER: _description_
        """
        contraste = np.std(imagen_gris)
        if contraste < 30:
            return cv2.MSER_create(delta=5, min_area=80, max_area=250000)
        elif contraste < 60:
            return cv2.MSER_create(delta=8, min_area=120, max_area=280000)
        else:
            return cv2.MSER_create(delta=12, min_area=140, max_area=300000)
    
    def _calcular_rango_hsv_adaptativo(self, imagen):
        """_summary_ Detecta niebla y ajusta el rango HSV para azul en consecuencia

        Args:
            imagen (np.ndarray): _description_

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        contraste = np.std(hsv[:, :, 2])
        brillo_medio = np.mean(hsv[:, :, 2])
        
        s_min = 50 if contraste < 40 else 100
        v_min = 40 if brillo_medio < 90 else 50
        
        azul_bajo = np.array([95, s_min, v_min])
        azul_alto = np.array([130, 255, 255])
        return azul_bajo, azul_alto
    
    def _score_geometrico_simple(self, recorte):
        """_summary_  Busca el marco del cartel usando Canny adaptativo y validamos con Hough 

        Args:
            recorte (np.ndarray): _description_

        Returns:
            float: _description_
        """
        h, w = recorte.shape[:2]
        if h < 20 or w < 20: return 0.0
        
        gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        # Difuminamos para que Canny no saque textura del asfalto o árboles
        gris_blur = cv2.GaussianBlur(gris, (3, 3), 0)
        
        # Calculamos la mediana de intensidad y ajustamos los umbrales dinámicamente
        mediana = np.median(gris_blur)
        lower = int(max(0, (1.0 - 0.33) * mediana))
        upper = int(min(255, (1.0 + 0.33) * mediana))
        bordes = cv2.Canny(gris_blur, lower, upper)
        
        min_line = min(w, h) * 0.25
        lines = cv2.HoughLinesP(bordes, 1, np.pi / 180, 20, minLineLength=int(min_line), maxLineGap=10)
        
        if lines is None: return 0.2
        
        h_lines, v_lines = 0, 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            # Detección de líneas horizontales (margen de 20 grados)
            if angle < 20 or angle > 160: h_lines += 1
            # Detección de líneas verticales (margen entre 70 y 110 grados)
            elif 70 < angle < 110: v_lines += 1
        
        # Asignación de score geométrico
        if h_lines >= 1 and v_lines >= 1: return 1.0
        elif h_lines >= 2 or v_lines >= 2: return 0.7
        else: return 0.3
    
    def detectar(self, imagen):
        """_summary_

        Args:
            imagen (np.ndarray): _description_

        Returns:
            List[Panel]: _description_
        """
        paneles_detectados = []
        azul_bajo, azul_alto = self._calcular_rango_hsv_adaptativo(imagen)
        
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        mser = self._get_mser_adaptive(gris)
        regiones, _ = mser.detectRegions(gris)
        
        alto_img, ancho_img = imagen.shape[:2]
        candidatos = []
        
        for region in regiones:
            x, y, w, h = cv2.boundingRect(region)
            if h < 20 or w < 25 or w > ancho_img * 0.75: continue
            
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 6.5:
                factor = 0.12 if w < 80 else 0.06
                mx, my = int(w * factor), int(h * factor)
                
                nx, ny = max(0, x - mx), max(0, y - my)
                nw, nh = min(ancho_img - nx, w + 2 * mx), min(alto_img - ny, h + 2 * my)
                candidatos.append([nx, ny, nx+nw, ny+nh])
        
        for x1, y1, x2, y2 in candidatos:
            recorte = imagen[y1:y2, x1:x2]
            if recorte.size == 0: continue
            
            recorte_resized = cv2.resize(recorte, (self.ancho, self.alto))
            hsv_resized = cv2.cvtColor(recorte_resized, cv2.COLOR_BGR2HSV)
            mascara_azul = cv2.inRange(hsv_resized, azul_bajo, azul_alto)
            
            # cierre para tapar agujeros por texto
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            mascara_azul = cv2.morphologyEx(mascara_azul, cv2.MORPH_CLOSE, kernel).astype(np.float32) / 255.0
            
            azul_centro = np.sum(mascara_azul[self.zona_interna])
            azul_borde = np.sum(mascara_azul[self.zona_externa])
            
            precision = azul_centro / (azul_centro + azul_borde + 1e-6)
            recall = azul_centro / np.sum(self.zona_interna)
            score_color = 2 * (precision * recall) / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0
            
            if score_color < 0.25: continue
            
            score_geo = self._score_geometrico_simple(recorte)
            
            # puntuación combinada
            score_final = 0.7 * score_color + 0.3 * score_geo
            
            if score_final > self.confidence_threshold:
                paneles_detectados.append(Panel(x1, y1, x2, y2, score_final))
        
        return self._eliminar_repetidas_avanzado(paneles_detectados)
    
    def _eliminar_repetidas_avanzado(self, lista_paneles):
        """_summary_

        Args:
            lista_paneles (List[Panel]): _description_

        Returns:
            List[Panel]: _description_
        """
        if not lista_paneles: return []
        
        lista_paneles.sort(key=lambda p: p.score, reverse=True)
        finales = []
        
        while lista_paneles:
            mejor = lista_paneles.pop(0)
            finales.append(mejor)
            restantes = []
            
            for p in lista_paneles:
                iz = max(mejor.x1, p.x1)
                ar = max(mejor.y1, p.y1)
                de = min(mejor.x2, p.x2)
                ab = min(mejor.y2, p.y2)
                inter = max(0, de - iz + 1) * max(0, ab - ar + 1)
                
                area_mejor = (mejor.x2 - mejor.x1 + 1) * (mejor.y2 - mejor.y1 + 1)
                area_p = (p.x2 - p.x1 + 1) * (p.y2 - p.y1 + 1)
                
                iou = inter / float(area_mejor + area_p - inter + 1e-6)
                
                # quitar cajas pequeñas contenidas dentro de grandes
                iom = inter / float(min(area_mejor, area_p) + 1e-6)
                
                # quitar cajas solapadas
                if iou <= 0.30 and iom <= 0.70:
                    restantes.append(p)
            
            lista_paneles = restantes
            
        return finales