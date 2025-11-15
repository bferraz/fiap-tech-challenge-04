"""
Coletor de estatísticas para análise de vídeo
"""
from collections import defaultdict, Counter
from typing import Dict, List, Any
import json


class StatisticsCollector:
    """Classe para coletar e gerenciar estatísticas durante a análise"""
    
    def __init__(self):
        """Inicializa o coletor de estatísticas"""
        # Estatísticas gerais
        self.total_frames = 0
        self.frames_with_faces = 0
        self.frames_with_poses = 0
        
        # Faces
        self.total_faces_count = 0  # Total de detecções de rostos
        self.face_detections_per_frame = []
        
        # Emoções
        self.emotions_counter = Counter()  # Contador geral de emoções
        self.emotions_timeline = []  # Lista de (timestamp, emotion)
        
        # Atividades
        self.activities_by_person = defaultdict(list)  # {person_id: [(timestamp, activity, duration)]}
        self.activities_counter = Counter()  # Contador geral de atividades
        
        # Anomalias
        self.anomalies = []  # Lista de (timestamp, frame, type, description)
        self.anomalies_by_type = Counter()
        
        # Timeline geral
        self.timeline = []  # Lista de eventos para o relatório
    
    def add_frame(self):
        """Incrementa o contador de frames"""
        self.total_frames += 1
    
    def add_face_detection(self, frame_number: int, num_faces: int):
        """
        Registra detecção de rostos em um frame
        
        Args:
            frame_number: Número do frame
            num_faces: Número de rostos detectados no frame
        """
        if num_faces > 0:
            self.frames_with_faces += 1
            self.total_faces_count += num_faces
        self.face_detections_per_frame.append(num_faces)
    
    def add_emotion(self, timestamp: float, emotion: str):
        """
        Registra uma emoção detectada
        
        Args:
            timestamp: Timestamp em segundos
            emotion: Emoção detectada
        """
        self.emotions_counter[emotion] += 1
        self.emotions_timeline.append((timestamp, emotion))
    
    def add_activity(self, timestamp: float, person_id: str, activity: str, duration: float = 0):
        """
        Registra uma atividade detectada
        
        Args:
            timestamp: Timestamp em segundos
            person_id: ID da pessoa
            activity: Atividade detectada
            duration: Duração da atividade em segundos
        """
        self.activities_by_person[person_id].append((timestamp, activity, duration))
        self.activities_counter[activity] += 1
    
    def add_pose_detection(self):
        """Incrementa o contador de frames com poses detectadas"""
        self.frames_with_poses += 1
    
    def add_anomaly(self, timestamp: float, frame_number: int, anomaly_type: str, description: str):
        """
        Registra uma anomalia detectada
        
        Args:
            timestamp: Timestamp em segundos
            frame_number: Número do frame
            anomaly_type: Tipo de anomalia
            description: Descrição da anomalia
        """
        self.anomalies.append({
            'timestamp': timestamp,
            'frame': frame_number,
            'type': anomaly_type,
            'description': description
        })
        self.anomalies_by_type[anomaly_type] += 1
    
    def add_timeline_event(self, timestamp: float, event: str):
        """
        Adiciona um evento à timeline
        
        Args:
            timestamp: Timestamp em segundos
            event: Descrição do evento
        """
        self.timeline.append((timestamp, event))
    
    def get_overall_emotion_distribution(self) -> Dict[str, int]:
        """
        Retorna a distribuição geral de emoções
        
        Returns:
            Dicionário com contagem de cada emoção
        """
        return dict(self.emotions_counter)
    
    def get_dominant_activity(self) -> str:
        """
        Retorna a atividade mais comum no vídeo
        
        Returns:
            Atividade dominante
        """
        if self.activities_counter:
            return self.activities_counter.most_common(1)[0][0]
        return "unknown"
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo completo das estatísticas
        
        Returns:
            Dicionário com todas as estatísticas
        """
        emotion_dist = self.get_overall_emotion_distribution()
        total_emotions = sum(emotion_dist.values())
        
        # Calcular top 5 emoções
        top_5_emotions = []
        if emotion_dist:
            sorted_emotions = sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            top_5_emotions = [(emotion, f"{(count / total_emotions * 100):.2f}%") for emotion, count in sorted_emotions]
        
        return {
            'general': {
                'total_frames': self.total_frames,
                'frames_with_faces': self.frames_with_faces,
                'frames_with_poses': self.frames_with_poses,
            },
            'faces': {
                'detection_rate': f"{(self.frames_with_faces / self.total_frames * 100):.2f}%" if self.total_frames > 0 else "0%",
            },
            'emotions': {
                'distribution': {k: f"{(v / total_emotions * 100):.2f}%" for k, v in emotion_dist.items()} if total_emotions > 0 else {},
                'dominant': max(emotion_dist.items(), key=lambda x: x[1])[0] if emotion_dist else "unknown",
                'top_5': top_5_emotions,
            },
            'activities': {
                'distribution': dict(self.activities_counter),
                'dominant': self.get_dominant_activity(),
            },
            'anomalies': {
                'total': len(self.anomalies),
                'by_type': dict(self.anomalies_by_type),
                'details': self.anomalies,
            },
        }
    
    def export_to_json(self, filepath: str):
        """
        Exporta as estatísticas para um arquivo JSON
        
        Args:
            filepath: Caminho do arquivo de saída
        """
        summary = self.get_summary()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
