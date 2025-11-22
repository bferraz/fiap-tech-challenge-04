"""
Configurações gerais do projeto de análise de vídeo
"""

# Configurações de vídeo
VIDEO_CONFIG = {
    'input_path': 'video.mp4',
    'output_path': 'output/video_processado.mp4',
    'codec': 'mp4v',
    'process_every_n_frames': 1,  # Processar todos os frames (1) ou pular frames (2, 3, etc)
}

# Configurações de detecção facial
FACE_CONFIG = {
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'min_size': (30, 30),
    'detection_method': 'hog',  # 'hog' (rápido) ou 'cnn' (preciso, mas lento)
    'face_tracking_threshold': 50,  # Distância máxima para considerar o mesmo rosto
}

# Configurações de análise de emoções
EMOTION_CONFIG = {
    'enforce_detection': False,
    'detector_backend': 'opencv',
    'emotions': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
}

# Configurações de detecção de atividades
ACTIVITY_CONFIG = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'activities': {
        'standing': 'Em pé',
        'sitting': 'Sentado',
        'arms_up': 'Braços levantados',
        'crouching': 'Agachado',
        'leaning': 'Inclinado',
        'walking': 'Caminhando',
        'waving': 'Acenando',
    }
}

# Configurações de detecção de anomalias
ANOMALY_CONFIG = {
    'sudden_movement_threshold': 3.0,  # Multiplicador da velocidade média
    'pose_confidence_threshold': 0.5,
    'emotion_change_time_threshold': 2.0,  # Segundos
    'extreme_emotion_duration': 5.0,  # Segundos
}

# Configurações de visualização
VISUALIZATION_CONFIG = {
    'show_face_boxes': True,
    'show_emotions': True,
    'show_pose_landmarks': True,
    'show_activity_label': True,
    'show_anomaly_alerts': True,
    'show_stats_hud': True,
    'colors': {
        'face_box': (0, 255, 0),
        'emotion_label': (36, 255, 12),
        'pose_landmarks': (0, 255, 255),
        'anomaly_alert': (0, 0, 255),
        'hud_text': (255, 255, 255),
    },
    'font': 0,  # cv2.FONT_HERSHEY_SIMPLEX
    'font_scale': 0.6,
    'font_thickness': 2,
}

# Configurações de relatório
REPORT_CONFIG = {
    'output_path': 'output/relatorio.txt',
    'json_output_path': 'output/relatorio.json',
    'include_timeline': True,
    'timeline_interval': 5,  # Segundos entre entradas da timeline
}
