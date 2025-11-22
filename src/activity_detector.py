"""
Detector de atividades usando MediaPipe Pose
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import mediapipe as mp
except ImportError:
    print("AVISO: MediaPipe não instalado. Instale com: pip install mediapipe")


class ActivityDetector:
    """Classe para detectar poses e classificar atividades"""
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Inicializa o detector de atividades
        
        Args:
            min_detection_confidence: Confiança mínima para detecção
            min_tracking_confidence: Confiança mínima para rastreamento
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Histórico de poses para detecção de movimento
        self.pose_history = []
        self.history_size = 10
    
    def detect_pose(self, frame: np.ndarray):
        """
        Detecta a pose no frame
        
        Args:
            frame: Frame do vídeo em BGR
        
        Returns:
            Resultado da detecção do MediaPipe
        """
        # Converter para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar o frame
        results = self.pose.process(rgb_frame)
        
        return results
    
    def draw_pose_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Desenha os landmarks da pose no frame
        
        Args:
            frame: Frame do vídeo
            results: Resultado da detecção de pose
        
        Returns:
            Frame com landmarks desenhados
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        return frame
    
    def classify_activity(self, landmarks) -> str:
        """
        Classifica a atividade com base nos landmarks
        
        Args:
            landmarks: Landmarks da pose do MediaPipe
        
        Returns:
            Nome da atividade detectada
        """
        if not landmarks:
            return 'unknown'
        
        # Extrair landmarks importantes
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        
        # Calcular altura média dos ombros e quadris
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        knee_y = (left_knee.y + right_knee.y) / 2
        
        # Verificar braços levantados
        arms_up = (left_elbow.y < shoulder_y and right_elbow.y < shoulder_y) or \
                  (left_wrist.y < shoulder_y and right_wrist.y < shoulder_y)
        
        if arms_up:
            return 'arms_up'
        
        # Verificar acenando (braço levantado unilateral)
        left_arm_up = left_wrist.y < left_shoulder.y and left_elbow.y < left_shoulder.y
        right_arm_up = right_wrist.y < right_shoulder.y and right_elbow.y < right_shoulder.y
        
        if (left_arm_up and not right_arm_up) or (right_arm_up and not left_arm_up):
            return 'waving'
        
        # Verificar agachado
        if knee_y > hip_y + 0.1 and hip_y > shoulder_y + 0.1:
            return 'crouching'
        
        # Verificar sentado
        if abs(hip_y - knee_y) < 0.15:
            return 'sitting'
        
        # Verificar inclinado
        torso_angle = abs(shoulder_y - hip_y)
        if torso_angle < 0.15:
            return 'leaning'
        
        # Detectar movimento (caminhando)
        if len(self.pose_history) > 0:
            prev_hip_x = self.pose_history[-1]
            current_hip_x = (left_hip.x + right_hip.x) / 2
            movement = abs(current_hip_x - prev_hip_x)
            
            if movement > 0.02:
                return 'walking'
        
        # Atualizar histórico
        self.pose_history.append((left_hip.x + right_hip.x) / 2)
        if len(self.pose_history) > self.history_size:
            self.pose_history.pop(0)
        
        # Padrão: em pé
        return 'standing'
    
    def get_activity_translation(self, activity: str) -> str:
        """
        Traduz o nome da atividade para português
        
        Args:
            activity: Nome da atividade em inglês
        
        Returns:
            Nome da atividade em português
        """
        translations = {
            'standing': 'Em pé',
            'sitting': 'Sentado',
            'arms_up': 'Braços levantados',
            'crouching': 'Agachado',
            'leaning': 'Inclinado',
            'walking': 'Caminhando',
            'waving': 'Acenando',
            'unknown': 'Desconhecido'
        }
        return translations.get(activity, activity)
    
    def draw_activity_label(self, frame: np.ndarray, activity: str, 
                           position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Desenha o label da atividade no frame
        
        Args:
            frame: Frame do vídeo
            activity: Nome da atividade
            position: Posição do texto (x, y)
        
        Returns:
            Frame com o label desenhado
        """
        activity_text = self.get_activity_translation(activity)
        text = f"Atividade: {activity_text}"
        
        # Desenhar fundo
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(
            frame,
            (position[0] - 5, position[1] - text_size[1] - 5),
            (position[0] + text_size[0] + 5, position[1] + 5),
            (0, 0, 0),
            -1
        )
        
        # Desenhar texto
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        return frame
    
    def detect_and_classify(self, frame: np.ndarray) -> Tuple[any, str]:
        """
        Detecta pose e classifica atividade
        
        Args:
            frame: Frame do vídeo
        
        Returns:
            Tuple (results, activity)
        """
        results = self.detect_pose(frame)
        activity = 'unknown'
        
        if results.pose_landmarks:
            activity = self.classify_activity(results.pose_landmarks.landmark)
        
        return results, activity
    
    def calculate_movement_speed(self, landmarks) -> float:
        """
        Calcula a velocidade de movimento baseada nos landmarks
        
        Args:
            landmarks: Landmarks da pose
        
        Returns:
            Velocidade de movimento (0-1)
        """
        if not landmarks or len(self.pose_history) == 0:
            return 0.0
        
        # Calcular mudança de posição dos principais landmarks
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        current_center = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        
        if len(self.pose_history) > 0:
            prev_center = self.pose_history[-1]
            if isinstance(prev_center, tuple):
                distance = np.sqrt(
                    (current_center[0] - prev_center[0])**2 + 
                    (current_center[1] - prev_center[1])**2
                )
                return distance
        
        return 0.0
    
    def release(self):
        """Libera os recursos do MediaPipe"""
        self.pose.close()
