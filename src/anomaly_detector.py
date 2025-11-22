"""
Detector de anomalias em vídeos
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque


class AnomalyDetector:
    """Classe para detectar comportamentos anômalos"""
    
    def __init__(self, sudden_movement_threshold: float = 3.0, 
                 pose_confidence_threshold: float = 0.5,
                 emotion_change_time_threshold: float = 2.0):
        """
        Inicializa o detector de anomalias
        
        Args:
            sudden_movement_threshold: Multiplicador da velocidade média para detectar movimento brusco
            pose_confidence_threshold: Confiança mínima da pose para não ser considerada anômala
            emotion_change_time_threshold: Tempo mínimo entre mudanças emocionais (segundos)
        """
        self.sudden_movement_threshold = sudden_movement_threshold
        self.pose_confidence_threshold = pose_confidence_threshold
        self.emotion_change_time_threshold = emotion_change_time_threshold
        
        # Histórico de movimentos
        self.movement_history = deque(maxlen=30)  # Últimos 30 frames
        self.movement_speeds = []
        
        # Histórico de emoções por rosto
        self.emotion_history = {}  # {face_id: [(timestamp, emotion)]}
        
        # Histórico de poses
        self.pose_history = []
        
    def detect_sudden_movement(self, movement_speed: float) -> bool:
        """
        Detecta movimento brusco baseado na velocidade
        
        Args:
            movement_speed: Velocidade de movimento atual
        
        Returns:
            True se movimento brusco detectado
        """
        self.movement_speeds.append(movement_speed)
        
        # Precisa de pelo menos 10 frames para calcular média
        if len(self.movement_speeds) < 10:
            return False
        
        # Calcular velocidade média
        avg_speed = np.mean(self.movement_speeds[-30:])  # Média dos últimos 30 frames
        
        # Detectar movimento brusco
        if avg_speed > 0 and movement_speed > avg_speed * self.sudden_movement_threshold:
            return True
        
        return False
    
    def detect_abnormal_pose(self, landmarks, min_confidence: float) -> Tuple[bool, str]:
        """
        Detecta pose anômala baseada na confiança dos landmarks
        
        Args:
            landmarks: Landmarks da pose do MediaPipe
            min_confidence: Confiança mínima detectada
        
        Returns:
            Tuple (is_abnormal, reason)
        """
        if not landmarks:
            return False, ""
        
        # Verificar confiança baixa
        if min_confidence < self.pose_confidence_threshold:
            return True, f"Confiança baixa da pose ({min_confidence:.2f})"
        
        # Verificar ângulos corporais incomuns
        # Aqui podemos adicionar mais lógica específica
        # Por exemplo, verificar se membros estão em posições impossíveis
        
        return False, ""
    
    def detect_rapid_emotion_change(self, timestamp: float, emotion: str) -> bool:
        """
        Detecta mudanças emocionais muito rápidas
        
        Args:
            timestamp: Timestamp atual em segundos
            emotion: Emoção atual
        
        Returns:
            True se mudança emocional rápida detectada
        """
        # Usar histórico geral (não por rosto)
        if 'general' not in self.emotion_history:
            self.emotion_history['general'] = []
        
        history = self.emotion_history['general']
        
        # Adicionar emoção atual ao histórico
        history.append((timestamp, emotion))
        
        # Manter apenas últimos 10 registros
        if len(history) > 10:
            history.pop(0)
        
        # Precisa de pelo menos 2 registros para comparar
        if len(history) < 2:
            return False
        
        # Verificar mudança recente
        prev_timestamp, prev_emotion = history[-2]
        time_diff = timestamp - prev_timestamp
        
        # Se emoção mudou em menos que o threshold
        if prev_emotion != emotion and time_diff < self.emotion_change_time_threshold:
            return True
        
        return False
    
    def detect_extreme_emotion_sustained(self, emotion: str, 
                                        current_timestamp: float, duration_threshold: float = 5.0) -> bool:
        """
        Detecta emoções extremas sustentadas por muito tempo
        
        Args:
            emotion: Emoção atual
            current_timestamp: Timestamp atual
            duration_threshold: Duração mínima para considerar anômalo (segundos)
        
        Returns:
            True se emoção extrema sustentada detectada
        """
        extreme_emotions = ['angry', 'fear', 'disgust']
        
        if emotion not in extreme_emotions:
            return False
        
        if 'general' not in self.emotion_history:
            return False
        
        history = self.emotion_history['general']
        
        # Verificar quanto tempo a emoção extrema está presente
        duration = 0
        for i in range(len(history) - 1, -1, -1):
            ts, emo = history[i]
            if emo != emotion:
                break
            duration = current_timestamp - ts
        
        return duration >= duration_threshold
    
    def analyze_frame_for_anomalies(self, frame_number: int, timestamp: float,
                                   movement_speed: float,
                                   pose_landmarks,
                                   pose_confidence: float,
                                   emotions_data: Dict[str, Dict]) -> List[Dict]:
        """
        Analisa um frame completo para detectar anomalias
        
        Args:
            frame_number: Número do frame
            timestamp: Timestamp em segundos
            movement_speed: Velocidade de movimento
            pose_landmarks: Landmarks da pose
            pose_confidence: Confiança da detecção de pose
            emotions_data: Dados de emoções {face_id: {'emotion': str, ...}}
        
        Returns:
            Lista de anomalias detectadas
        """
        anomalies = []
        
        # 1. Detectar movimento brusco
        if self.detect_sudden_movement(movement_speed):
            anomalies.append({
                'frame': frame_number,
                'timestamp': timestamp,
                'type': 'sudden_movement',
                'description': f'Movimento brusco detectado (velocidade: {movement_speed:.4f})'
            })
        
        # 2. Detectar pose anômala
        if pose_landmarks:
            is_abnormal, reason = self.detect_abnormal_pose(pose_landmarks, pose_confidence)
            if is_abnormal:
                anomalies.append({
                    'frame': frame_number,
                    'timestamp': timestamp,
                    'type': 'abnormal_pose',
                    'description': f'Pose anômala: {reason}'
                })
        
        # 3. Detectar mudanças emocionais rápidas e emoções extremas
        for face_key, emotion_info in emotions_data.items():
            emotion = emotion_info.get('emotion', 'neutral')
            
            if self.detect_rapid_emotion_change(timestamp, emotion):
                anomalies.append({
                    'frame': frame_number,
                    'timestamp': timestamp,
                    'type': 'rapid_emotion_change',
                    'description': f'Mudança emocional rápida para {emotion}'
                })
            
            # 4. Detectar emoções extremas sustentadas
            if self.detect_extreme_emotion_sustained(emotion, timestamp):
                anomalies.append({
                    'frame': frame_number,
                    'timestamp': timestamp,
                    'type': 'sustained_extreme_emotion',
                    'description': f'Emoção extrema ({emotion}) sustentada'
                })
        
        return anomalies
    
    def get_average_movement_speed(self) -> float:
        """
        Retorna a velocidade média de movimento
        
        Returns:
            Velocidade média
        """
        if not self.movement_speeds:
            return 0.0
        return np.mean(self.movement_speeds)
    
    def reset(self):
        """Reseta o detector de anomalias"""
        self.movement_history.clear()
        self.movement_speeds.clear()
        self.emotion_history.clear()
        self.pose_history.clear()
