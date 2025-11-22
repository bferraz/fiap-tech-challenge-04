"""
Detector de rostos sem diferenciação individual
Usa MediaPipe Face Detection para maior robustez
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
import mediapipe as mp


class FaceDetector:
    """Classe para detectar rostos no vídeo usando MediaPipe"""
    
    def __init__(self, tracking_threshold: int = 50):
        """
        Inicializa o detector de rostos
        
        Args:
            tracking_threshold: Parâmetro não usado, mantido para compatibilidade
        """
        # Inicializar MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = modelo completo (melhor para rostos distantes)
            min_detection_confidence=0.6  # Aumentado para reduzir falsos positivos
        )
        
        # Inicializar MediaPipe Face Mesh para validação adicional
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Fallback: Haar Cascade para casos extremos
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        
        # Haar Cascade para olhos (validação adicional)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Contador de rostos detectados
        self.total_faces_detected = 0
    
    def _validate_face_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Valida se uma região detectada é realmente um rosto usando múltiplas verificações
        
        Args:
            frame: Frame do vídeo
            bbox: Bounding box da região (x, y, w, h)
        
        Returns:
            True se a região é um rosto válido, False caso contrário
        """
        x, y, w, h = bbox
        
        # 1. Validação de tamanho mínimo
        if w < 20 or h < 20:
            return False
        
        # 2. Validação de proporção (rostos geralmente são retangulares verticais ou quadrados)
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 1.8:  # Muito largo ou muito alto
            return False
        
        # 3. Extrair região
        face_region = frame[y:y+h, x:x+w]
        
        # 4. Verificar cor da pele usando HSV (rostos têm tons de pele característicos)
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Máscaras para tons de pele (múltiplas faixas para diferentes tons)
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        
        lower_skin2 = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin2 = np.array([20, 150, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        skin_percentage = np.sum(skin_mask > 0) / (w * h)
        
        # Pelo menos 15% da região deve ter tom de pele
        if skin_percentage < 0.15:
            return False
        
        # 5. Verificar textura (rostos têm textura característica)
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Desvio padrão (rostos têm variação de intensidade moderada)
        std_dev = np.std(gray)
        if std_dev < 15 or std_dev > 80:  # Nem muito uniforme, nem muito ruidoso
            return False
        
        # 6. Verificar gradientes (rostos têm bordas/contornos característicos)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        if edge_density < 0.03 or edge_density > 0.4:  # Densidade de bordas moderada
            return False
        
        # 7. Tentar detectar olhos na região (forte indicador de rosto)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(int(w * 0.1), int(h * 0.1)),  # Olhos são ~10% do rosto
            maxSize=(int(w * 0.5), int(h * 0.5))
        )
        
        # Se detectou 1 ou 2 olhos, é muito provável que seja um rosto
        if len(eyes) >= 1:
            return True
        
        # 8. Validação com Face Mesh (verificar se tem landmarks faciais)
        rgb_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(rgb_region)
        
        if mesh_results.multi_face_landmarks:
            # Se Face Mesh detectou landmarks, é definitivamente um rosto
            return True
        
        # Se passou pelas validações básicas mas não detectou olhos nem landmarks,
        # aceitar se tiver boa pontuação nos critérios anteriores
        if skin_percentage > 0.25 and 20 < std_dev < 70 and 0.05 < edge_density < 0.35:
            return True
        
        return False
    
    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostos usando MediaPipe Face Detection com validação adicional
        
        Args:
            frame: Frame do vídeo
        
        Returns:
            Lista de tuplas (x, y, w, h) com as coordenadas dos rostos validados
        """
        # Converter BGR para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar imagem
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            
            for detection in results.detections:
                # Verificar score de confiança
                score = detection.score[0] if hasattr(detection, 'score') else 1.0
                
                # Filtrar detecções com confiança muito baixa
                if score < 0.6:
                    continue
                
                # Obter bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Converter coordenadas relativas para absolutas
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Garantir que as coordenadas estejam dentro dos limites
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Validar se é realmente um rosto (não mão, planta, etc)
                if self._validate_face_region(frame, (x, y, width, height)):
                    faces.append((x, y, width, height))
        
        return faces
    
    def detect_faces_haar_cascade(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostos usando Haar Cascade (frontal + perfil) como fallback
        
        Args:
            frame: Frame do vídeo
        
        Returns:
            Lista de tuplas (x, y, w, h) com as coordenadas dos rostos
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Melhorar contraste
        gray = cv2.equalizeHist(gray)
        
        h, w = frame.shape[:2]
        all_faces = []
        
        # Detecção frontal com parâmetros relaxados
        frontal_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,      # Mais escalas
            minNeighbors=3,        # Menos rigoroso
            minSize=(30, 30),      # Menor tamanho mínimo
            maxSize=(w, h),        # Sem limite máximo!
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(frontal_faces)
        
        # Detecção de perfil (esquerda)
        profile_faces = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            maxSize=(w, h),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(profile_faces)
        
        # Detecção de perfil (direita) - espelhar imagem
        gray_flipped = cv2.flip(gray, 1)
        profile_faces_flipped = self.profile_cascade.detectMultiScale(
            gray_flipped,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            maxSize=(w, h),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Ajustar coordenadas do flip
        for (x, y, fw, fh) in profile_faces_flipped:
            all_faces.append((w - x - fw, y, fw, fh))
        
        # Remover duplicatas usando Non-Maximum Suppression
        unique_faces = self._remove_duplicates(all_faces)
        
        return unique_faces
    
    def _remove_duplicates(self, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Remove rostos duplicados usando IoU (Intersection over Union)
        
        Args:
            faces: Lista de bounding boxes
        
        Returns:
            Lista de bounding boxes únicos
        """
        if not faces:
            return []
        
        unique_faces = []
        for face in faces:
            is_duplicate = False
            x1, y1, w1, h1 = face
            
            for existing in unique_faces:
                x2, y2, w2, h2 = existing
                
                # Calcular IoU
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.3:  # 30% de overlap = duplicata
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostos no frame usando abordagem híbrida
        Primeiro tenta MediaPipe, depois Haar Cascade como fallback
        
        Args:
            frame: Frame do vídeo
        
        Returns:
            Lista de tuplas (x, y, w, h) com as coordenadas dos rostos
        """
        # Tentar MediaPipe primeiro (mais robusto)
        faces = self.detect_faces_mediapipe(frame)
        
        # Se MediaPipe não detectou nada, usar Haar Cascade
        if not faces:
            faces = self.detect_faces_haar_cascade(frame)
        
        return faces
    
    def detect_and_track(self, frame: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Detecta rostos em um frame sem diferenciação individual
        
        Args:
            frame: Frame do vídeo
        
        Returns:
            Dicionário {'face': (x, y, w, h)} para cada rosto detectado
        """
        faces = self.detect_faces(frame)
        
        # Criar dicionário simples sem IDs únicos
        detected_faces = {}
        for i, face in enumerate(faces):
            detected_faces[f'face_{i}'] = face
        
        self.total_faces_detected = len(faces)
        return detected_faces
    
    def draw_faces(self, frame: np.ndarray, tracked_faces: Dict[str, Tuple[int, int, int, int]], 
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Desenha retângulos dos rostos no frame (sem labels de ID)
        
        Args:
            frame: Frame do vídeo
            tracked_faces: Dicionário com faces detectadas
            color: Cor do retângulo (BGR)
            thickness: Espessura da linha
        
        Returns:
            Frame com os rostos marcados
        """
        for face_key, (x, y, w, h) in tracked_faces.items():
            # Desenhar apenas retângulo, sem ID
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        return frame
    
    def get_face_image(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extrai a região do rosto do frame
        
        Args:
            frame: Frame completo
            bbox: Bounding box do rosto (x, y, w, h)
        
        Returns:
            Imagem recortada do rosto
        """
        x, y, w, h = bbox
        return frame[y:y+h, x:x+w]
    
    def reset_tracking(self):
        """Reseta o contador de rostos"""
        self.total_faces_detected = 0
    
    def __del__(self):
        """Libera recursos do MediaPipe"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
