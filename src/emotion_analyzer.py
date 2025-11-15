"""
Analisador de emoções usando DeepFace
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    from deepface import DeepFace
except ImportError:
    print("AVISO: DeepFace não instalado. Instale com: pip install deepface")


class EmotionAnalyzer:
    """Classe para analisar emoções em rostos detectados"""
    
    def __init__(self, detector_backend: str = 'opencv', enforce_detection: bool = False):
        """
        Inicializa o analisador de emoções
        
        Args:
            detector_backend: Backend de detecção ('opencv', 'ssd', 'mtcnn', etc)
            enforce_detection: Se True, lança erro quando não detectar rosto
        """
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Vermelho
            'disgust': (0, 255, 255),  # Amarelo
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 0),      # Verde
            'sad': (255, 0, 0),        # Azul
            'surprise': (255, 255, 0), # Ciano
            'neutral': (200, 200, 200) # Cinza
        }
    
    def analyze_emotion(self, frame: np.ndarray, face_region: Optional[Tuple[int, int, int, int]] = None) -> List[Dict]:
        """
        Analisa emoções no frame
        
        Args:
            frame: Frame do vídeo
            face_region: Região específica do rosto (x, y, w, h) - opcional
        
        Returns:
            Lista de dicionários com informações de emoção
        """
        try:
            # Se uma região específica foi fornecida, usar apenas ela
            if face_region:
                x, y, w, h = face_region
                face_img = frame[y:y+h, x:x+w]
                result = DeepFace.analyze(
                    face_img, 
                    actions=['emotion'],
                    enforce_detection=self.enforce_detection,
                    detector_backend=self.detector_backend,
                    silent=True
                )
            else:
                result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'],
                    enforce_detection=self.enforce_detection,
                    detector_backend=self.detector_backend,
                    silent=True
                )
            
            # Garantir que result seja sempre uma lista
            if not isinstance(result, list):
                result = [result]
            
            return result
        except Exception as e:
            # Retornar lista vazia em caso de erro
            return []
    
    def get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> str:
        """
        Retorna a emoção dominante dado um dicionário de scores
        
        Args:
            emotion_scores: Dicionário com scores de emoções
        
        Returns:
            Nome da emoção dominante
        """
        if not emotion_scores:
            return 'neutral'
        return max(emotion_scores.items(), key=lambda x: x[1])[0]
    
    def draw_emotion(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                     emotion: str, confidence: float = 0) -> np.ndarray:
        """
        Desenha a emoção detectada no frame
        
        Args:
            frame: Frame do vídeo
            face_bbox: Bounding box do rosto (x, y, w, h)
            emotion: Emoção detectada
            confidence: Confiança da detecção (0-100)
        
        Returns:
            Frame com a emoção desenhada
        """
        x, y, w, h = face_bbox
        
        # Cor baseada na emoção
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Preparar texto
        if confidence > 0:
            text = f"{emotion} ({confidence:.1f}%)"
        else:
            text = emotion
        
        # Posição do texto (acima do rosto)
        text_x = x
        text_y = y + h + 20
        
        # Desenhar fundo do texto
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            frame, 
            (text_x, text_y - text_size[1] - 5), 
            (text_x + text_size[0] + 5, text_y + 5), 
            (0, 0, 0), 
            -1
        )
        
        # Desenhar texto
        cv2.putText(
            frame, 
            text, 
            (text_x + 2, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            color, 
            2
        )
        
        return frame
    
    def analyze_faces_emotions(self, frame: np.ndarray, 
                               tracked_faces: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, Dict]:
        """
        Analisa emoções de múltiplos rostos rastreados
        
        Args:
            frame: Frame do vídeo
            tracked_faces: Dicionário {face_id: (x, y, w, h)}
        
        Returns:
            Dicionário {face_id: {'emotion': str, 'confidence': float, 'scores': dict}}
        """
        emotions_data = {}
        
        for face_id, bbox in tracked_faces.items():
            try:
                x, y, w, h = bbox
                # Garantir que as coordenadas estejam dentro dos limites
                h_frame, w_frame = frame.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, w_frame - x)
                h = min(h, h_frame - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                # Analisar emoção
                result = self.analyze_emotion(frame, (x, y, w, h))
                
                if result:
                    emotion_data = result[0].get('emotion', {})
                    dominant_emotion = self.get_dominant_emotion(emotion_data)
                    confidence = emotion_data.get(dominant_emotion, 0)
                    
                    emotions_data[face_id] = {
                        'emotion': dominant_emotion,
                        'confidence': confidence,
                        'scores': emotion_data
                    }
            except Exception as e:
                # Em caso de erro, usar emoção neutra
                emotions_data[face_id] = {
                    'emotion': 'neutral',
                    'confidence': 0,
                    'scores': {}
                }
        
        return emotions_data
    
    def draw_emotions_on_frame(self, frame: np.ndarray, 
                               tracked_faces: Dict[str, Tuple[int, int, int, int]],
                               emotions_data: Dict[str, Dict]) -> np.ndarray:
        """
        Desenha todas as emoções detectadas no frame
        
        Args:
            frame: Frame do vídeo
            tracked_faces: Dicionário com faces rastreadas
            emotions_data: Dicionário com dados de emoções
        
        Returns:
            Frame com as emoções desenhadas
        """
        for face_id, bbox in tracked_faces.items():
            if face_id in emotions_data:
                emotion_info = emotions_data[face_id]
                frame = self.draw_emotion(
                    frame, 
                    bbox, 
                    emotion_info['emotion'], 
                    emotion_info['confidence']
                )
        
        return frame
