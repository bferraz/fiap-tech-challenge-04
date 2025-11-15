"""
Analisador de vídeo principal - Orquestrador
"""
import cv2
import sys
import os
from tqdm import tqdm

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_detector import FaceDetector
from src.emotion_analyzer import EmotionAnalyzer
from src.activity_detector import ActivityDetector
from src.anomaly_detector import AnomalyDetector
from src.report_generator import ReportGenerator
from utils.video_processor import VideoProcessor
from utils.statistics_collector import StatisticsCollector
from config.settings import *


class VideoAnalyzer:
    """Classe principal que integra todos os módulos de análise"""
    
    def __init__(self, video_path: str, output_path: str = None):
        """
        Inicializa o analisador de vídeo
        
        Args:
            video_path: Caminho do vídeo de entrada
            output_path: Caminho do vídeo de saída
        """
        self.video_path = video_path
        self.output_path = output_path or VIDEO_CONFIG['output_path']
        
        # Inicializar módulos
        print("Inicializando módulos...")
        self.face_detector = FaceDetector(
            tracking_threshold=FACE_CONFIG['face_tracking_threshold']
        )
        self.emotion_analyzer = EmotionAnalyzer(
            detector_backend=EMOTION_CONFIG['detector_backend'],
            enforce_detection=EMOTION_CONFIG['enforce_detection']
        )
        self.activity_detector = ActivityDetector(
            min_detection_confidence=ACTIVITY_CONFIG['min_detection_confidence'],
            min_tracking_confidence=ACTIVITY_CONFIG['min_tracking_confidence']
        )
        self.anomaly_detector = AnomalyDetector(
            sudden_movement_threshold=ANOMALY_CONFIG['sudden_movement_threshold'],
            pose_confidence_threshold=ANOMALY_CONFIG['pose_confidence_threshold'],
            emotion_change_time_threshold=ANOMALY_CONFIG['emotion_change_time_threshold']
        )
        self.stats_collector = StatisticsCollector()
        self.report_generator = ReportGenerator(self.stats_collector)
        
        print("✓ Módulos inicializados com sucesso!")
    
    def draw_hud(self, frame, frame_number: int, timestamp: float, 
                 num_faces: int, activity: str, anomalies_count: int):
        """
        Desenha HUD com informações em tempo real
        
        Args:
            frame: Frame do vídeo
            frame_number: Número do frame
            timestamp: Timestamp em segundos
            num_faces: Número de rostos detectados
            activity: Atividade atual
            anomalies_count: Número total de anomalias
        """
        if not VISUALIZATION_CONFIG['show_stats_hud']:
            return frame
        
        hud_y = 30
        hud_x = 10
        line_height = 25
        
        hud_info = [
            f"Frame: {frame_number}",
            f"Tempo: {VideoProcessor.format_timestamp(timestamp)}",
            f"Rostos: {num_faces}",
            f"Atividade: {activity}",
            f"Anomalias: {anomalies_count}"
        ]
        
        # Desenhar fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x - 5, hud_y - 20), 
                     (300, hud_y + len(hud_info) * line_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Desenhar texto
        for i, text in enumerate(hud_info):
            y_pos = hud_y + i * line_height
            cv2.putText(frame, text, (hud_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       VISUALIZATION_CONFIG['colors']['hud_text'], 2)
        
        return frame
    
    def process_video(self):
        """Processa o vídeo completo"""
        print(f"\nProcessando vídeo: {self.video_path}")
        
        # Abrir vídeo
        video_processor = VideoProcessor(self.video_path)
        
        # Criar VideoWriter
        output_writer = VideoProcessor.create_video_writer(
            self.output_path,
            video_processor.fps,
            video_processor.width,
            video_processor.height,
            VIDEO_CONFIG['codec']
        )
        
        print(f"Resolução: {video_processor.width}x{video_processor.height}")
        print(f"FPS: {video_processor.fps:.2f}")
        print(f"Total de frames: {video_processor.total_frames}")
        print(f"Duração: {VideoProcessor.format_timestamp(video_processor.duration)}")
        print("\nIniciando análise...\n")
        
        # Informações do vídeo para o relatório
        video_info = {
            'path': self.video_path,
            'width': video_processor.width,
            'height': video_processor.height,
            'fps': video_processor.fps,
            'total_frames': video_processor.total_frames,
            'duration': video_processor.duration
        }
        
        # Processar frames
        with tqdm(total=video_processor.total_frames, desc="Analisando vídeo") as pbar:
            frame_number = 0
            
            while True:
                ret, frame = video_processor.read_frame()
                
                if not ret:
                    break
                
                frame_number += 1
                timestamp = video_processor.get_timestamp()
                
                # Adicionar frame às estatísticas
                self.stats_collector.add_frame()
                
                # 1. Detectar rostos
                tracked_faces = self.face_detector.detect_and_track(frame)
                num_faces = len(tracked_faces)
                
                # Registrar detecção de rostos
                self.stats_collector.add_face_detection(frame_number, num_faces)
                
                # Desenhar rostos
                if VISUALIZATION_CONFIG['show_face_boxes'] and tracked_faces:
                    frame = self.face_detector.draw_faces(
                        frame, tracked_faces, 
                        VISUALIZATION_CONFIG['colors']['face_box']
                    )
                
                # 2. Analisar emoções
                emotions_data = {}
                if tracked_faces and VISUALIZATION_CONFIG['show_emotions']:
                    emotions_data = self.emotion_analyzer.analyze_faces_emotions(frame, tracked_faces)
                    
                    # Registrar emoções (sem diferenciar por rosto)
                    for face_key, emotion_info in emotions_data.items():
                        self.stats_collector.add_emotion(
                            timestamp, emotion_info['emotion']
                        )
                    
                    # Desenhar emoções
                    frame = self.emotion_analyzer.draw_emotions_on_frame(
                        frame, tracked_faces, emotions_data
                    )
                
                # 3. Detectar pose e atividade
                pose_results, activity = self.activity_detector.detect_and_classify(frame)
                
                if pose_results.pose_landmarks:
                    self.stats_collector.add_pose_detection()
                    
                    # Registrar atividade
                    self.stats_collector.add_activity(timestamp, 'person_1', activity)
                    
                    # Desenhar pose
                    if VISUALIZATION_CONFIG['show_pose_landmarks']:
                        frame = self.activity_detector.draw_pose_landmarks(frame, pose_results)
                    
                    # Desenhar atividade
                    if VISUALIZATION_CONFIG['show_activity_label']:
                        frame = self.activity_detector.draw_activity_label(frame, activity)
                    
                    # Calcular velocidade de movimento
                    movement_speed = self.activity_detector.calculate_movement_speed(
                        pose_results.pose_landmarks.landmark
                    )
                else:
                    movement_speed = 0.0
                
                # 4. Detectar anomalias
                pose_confidence = 1.0 if pose_results.pose_landmarks else 0.0
                anomalies = self.anomaly_detector.analyze_frame_for_anomalies(
                    frame_number, timestamp, movement_speed,
                    pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None,
                    pose_confidence, emotions_data
                )
                
                # Registrar anomalias
                for anomaly in anomalies:
                    self.stats_collector.add_anomaly(
                        anomaly['timestamp'], anomaly['frame'],
                        anomaly['type'], anomaly['description']
                    )
                
                # Desenhar alerta de anomalia
                if anomalies and VISUALIZATION_CONFIG['show_anomaly_alerts']:
                    cv2.rectangle(frame, (0, 0), 
                                 (video_processor.width, video_processor.height),
                                 VISUALIZATION_CONFIG['colors']['anomaly_alert'], 10)
                    cv2.putText(frame, "! ANOMALIA DETECTADA !", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                               VISUALIZATION_CONFIG['colors']['anomaly_alert'], 3)
                
                # 5. Desenhar HUD
                frame = self.draw_hud(
                    frame, frame_number, timestamp,
                    len(tracked_faces), activity,
                    len(self.stats_collector.anomalies)
                )
                
                # Salvar frame processado
                output_writer.write(frame)
                
                # Atualizar barra de progresso
                pbar.update(1)
        
        # Liberar recursos
        video_processor.release()
        output_writer.release()
        self.activity_detector.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Vídeo processado salvo em: {self.output_path}")
        
        # Gerar relatórios
        print("\nGerando relatórios...")
        self.report_generator.generate_text_report(video_info, REPORT_CONFIG['output_path'])
        self.stats_collector.export_to_json(REPORT_CONFIG['json_output_path'])
        
        print("\n" + "=" * 80)
        print("ANÁLISE CONCLUÍDA!")
        print("=" * 80)
        
        # Exibir resumo
        summary_stats = self.report_generator.generate_summary_stats()
        print(f"\nRESUMO:")
        print(f"  • Total de frames analisados: {summary_stats['total_frames']}")
        print(f"  • Frames com rostos: {summary_stats.get('frames_with_faces', 0)}")
        
        # Mostrar top 3 emoções no resumo
        top_emotions = summary_stats.get('top_emotions', [])
        if top_emotions:
            print(f"  • Top 3 emoções:")
            for i, (emotion, percentage) in enumerate(top_emotions[:3], 1):
                emotion_pt = {'happy': 'Feliz', 'sad': 'Triste', 'angry': 'Raiva', 
                              'surprise': 'Surpresa', 'fear': 'Medo', 'disgust': 'Desgosto', 
                              'neutral': 'Neutro'}.get(emotion, emotion)
                print(f"    {i}. {emotion_pt}: {percentage}")
        
        print(f"  • Atividade principal: {summary_stats['dominant_activity']}")
        print(f"  • Total de anomalias detectadas: {summary_stats['total_anomalies']}")
        print(f"\n✓ Vídeo processado: {self.output_path}")
        print(f"✓ Relatório texto: {REPORT_CONFIG['output_path']}")
        print(f"✓ Relatório JSON: {REPORT_CONFIG['json_output_path']}")
        print()


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analisador de Vídeo - Tech Challenge')
    parser.add_argument('video', help='Caminho do vídeo de entrada')
    parser.add_argument('-o', '--output', help='Caminho do vídeo de saída', default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"ERRO: Vídeo não encontrado: {args.video}")
        return
    
    # Criar analisador e processar
    analyzer = VideoAnalyzer(args.video, args.output)
    analyzer.process_video()


if __name__ == "__main__":
    main()
