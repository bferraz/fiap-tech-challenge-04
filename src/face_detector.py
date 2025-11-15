"""
Detector de rostos sem diferenciação individual
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict


class FaceDetector:
    """Classe para detectar rostos no vídeo"""
    
    def __init__(self, tracking_threshold: int = 50):
        """
        Inicializa o detector de rostos
        
        Args:
            tracking_threshold: Parâmetro não usado, mantido para compatibilidade
        """
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Contador de rostos detectados
        self.total_faces_detected = 0
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostos no frame usando Haar Cascade com validação rigorosa
        
        Args:
            frame: Frame do vídeo
        
        Returns:
            Lista de tuplas (x, y, w, h) com as coordenadas dos rostos válidos
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_h, frame_w = frame.shape[:2]
        
        # Detecção MUITO RIGOROSA para evitar falsos positivos
        all_faces = []
        
        # Usar detecção com parâmetros MUITO rigorosos
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=10,   # MUITO rigoroso
            minSize=(60, 60),
            maxSize=(400, 400),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(faces)
        
        # Remover duplicatas (rostos detectados em ambas as passagens)
        unique_faces = []
        for face in all_faces:
            is_duplicate = False
            x1, y1, w1, h1 = face
            
            for existing in unique_faces:
                x2, y2, w2, h2 = existing
                
                # Calcular overlap (IoU - Intersection over Union)
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
                    
                    if iou > 0.5:  # 50% de overlap = duplicata
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        # Validar faces com critérios MUITO RIGOROSOS
        valid_faces = []
        for (x, y, w, h) in unique_faces:
            # 1. Proporção ESTRITA
            aspect_ratio = w / h
            if not (0.8 <= aspect_ratio <= 1.2):
                continue
            
            # 2. Área mínima grande
            area = w * h
            if area < 3600:  # Mínimo 60x60
                continue
            
            # 3. Não pode estar nas bordas
            if x < 20 or y < 20 or x + w > frame_w - 20 or y + h > frame_h - 20:
                continue
            
            # 4. Variação de intensidade alta
            roi_gray = gray[y:y+h, x:x+w]
            std_dev = np.std(roi_gray)
            if std_dev < 20:  # MUITO rigoroso
                continue
            
            # 5. Verificar gradientes (rostos têm bordas definidas)
            edges = cv2.Canny(roi_gray, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            if edge_density < 0.05 or edge_density > 0.5:  # Nem muito liso nem muito ruidoso
                continue
            
            valid_faces.append((x, y, w, h))
        
        return valid_faces
    
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
