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
        # MediaPipe Face Detection (encontra rostos em qualquer posição)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Modelo completo
            min_detection_confidence=0.5  # Moderado para não perder rostos
        )
        
        # MediaPipe Face Mesh (valida se é rosto real)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=False,  # Mais rápido
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Contador de rostos detectados
        self.total_faces_detected = 0
    
    
    def _is_valid_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Valida se uma detecção é realmente um rosto usando Face Mesh
        Face Mesh só retorna landmarks se for um rosto humano real
        
        Args:
            frame: Frame do vídeo
            bbox: Bounding box (x, y, w, h)
        
        Returns:
            True se é um rosto válido
        """
        x, y, w, h = bbox
        
        # Validações básicas rápidas
        if w < 30 or h < 30:
            return False
        
        # Proporção (rejeita objetos muito largos/altos)
        aspect_ratio = w / h
        if aspect_ratio < 0.4 or aspect_ratio > 2.0:
            return False
        
        # Extrair região do rosto
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + w, w_frame)
        y2 = min(y + h, h_frame)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        face_region = frame[y1:y2, x1:x2]
        
        # Validar com Face Mesh (definitivo)
        rgb_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(rgb_region)
        
        # Se Face Mesh detectou landmarks faciais, é um rosto real
        return mesh_results.multi_face_landmarks is not None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostos usando Face Detection + validação com Face Mesh
        
        Estratégia:
        1. Face Detection encontra candidatos (sensível, pega tudo)
        2. Face Mesh valida cada candidato (preciso, filtra falsos positivos)
        
        Args:
            frame: Frame do vídeo
        
        Returns:
            Lista de tuplas (x, y, w, h) com rostos validados
        """
        # Converter para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Detectar candidatos com Face Detection
        results = self.face_detection.process(rgb_frame)
        
        validated_faces = []
        
        if results.detections:
            h, w = frame.shape[:2]
            
            for detection in results.detections:
                # Score de confiança
                score = detection.score[0] if hasattr(detection, 'score') else 1.0
                
                # Filtrar detecções com confiança muito baixa (< 50%)
                if score < 0.5:
                    continue
                
                # Obter bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Converter para coordenadas absolutas
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Garantir coordenadas válidas
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # 2. Validar com Face Mesh (elimina falsos positivos)
                if self._is_valid_face(frame, (x, y, width, height)):
                    validated_faces.append((x, y, width, height))
        
        return validated_faces
    
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
