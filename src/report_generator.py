"""
Gerador de relatórios automáticos
"""
from typing import Dict, Any
from datetime import datetime
from utils.statistics_collector import StatisticsCollector
from utils.video_processor import VideoProcessor


class ReportGenerator:
    """Classe para gerar relatórios automáticos da análise"""
    
    def __init__(self, stats_collector: StatisticsCollector):
        """
        Inicializa o gerador de relatórios
        
        Args:
            stats_collector: Coletor de estatísticas
        """
        self.stats = stats_collector
    
    def format_timestamp(self, seconds: float) -> str:
        """
        Formata segundos para HH:MM:SS
        
        Args:
            seconds: Tempo em segundos
        
        Returns:
            String formatada
        """
        return VideoProcessor.format_timestamp(seconds)
    
    def generate_text_report(self, video_info: Dict[str, Any], output_path: str):
        """
        Gera relatório em formato texto
        
        Args:
            video_info: Informações do vídeo
            output_path: Caminho do arquivo de saída
        """
        summary = self.stats.get_summary()
        
        lines = []
        lines.append("=" * 80)
        lines.append("RELATÓRIO DE ANÁLISE DE VÍDEO")
        lines.append("=" * 80)
        lines.append(f"Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")
        lines.append("")
        
        # Informações Gerais
        lines.append("-" * 80)
        lines.append("INFORMAÇÕES GERAIS")
        lines.append("-" * 80)
        lines.append(f"Vídeo: {video_info.get('path', 'N/A')}")
        lines.append(f"Duração: {self.format_timestamp(video_info.get('duration', 0))}")
        lines.append(f"FPS: {video_info.get('fps', 0):.2f}")
        lines.append(f"Resolução: {video_info.get('width', 0)}x{video_info.get('height', 0)}")
        lines.append(f"Total de frames analisados: {summary['general']['total_frames']}")
        lines.append(f"Frames com rostos detectados: {summary['general']['frames_with_faces']}")
        lines.append(f"Frames com poses detectadas: {summary['general']['frames_with_poses']}")
        lines.append("")
        
        # Detecção Facial
        lines.append("-" * 80)
        lines.append("DETECÇÃO FACIAL")
        lines.append("-" * 80)
        lines.append(f"Frames com rostos detectados: {summary['general']['frames_with_faces']}")
        lines.append(f"Taxa de detecção: {summary['faces']['detection_rate']}")
        lines.append("")
        
        # Análise de Emoções
        lines.append("-" * 80)
        lines.append("ANÁLISE DE EMOÇÕES")
        lines.append("-" * 80)
        
        # Tradução de emoções
        emotion_translations = {
            'happy': 'Feliz',
            'sad': 'Triste',
            'angry': 'Raiva',
            'surprise': 'Surpresa',
            'fear': 'Medo',
            'disgust': 'Desgosto',
            'neutral': 'Neutro'
        }
        
        emotion_dist = summary['emotions']['distribution']
        if emotion_dist:
            # Obter top 5 emoções
            top_emotions = summary['emotions'].get('top_5', [])
            
            lines.append("TOP 5 EMOÇÕES DOMINANTES:")
            if top_emotions:
                for i, (emotion, percentage) in enumerate(top_emotions, 1):
                    emotion_pt = emotion_translations.get(emotion, emotion)
                    lines.append(f"  {i}. {emotion_pt}: {percentage}")
            else:
                lines.append("  Dados insuficientes")
            
            lines.append("")
            lines.append("Distribuição completa de emoções:")
            
            for emotion, percentage in sorted(emotion_dist.items(), key=lambda x: float(x[1].rstrip('%')), reverse=True):
                emotion_pt = emotion_translations.get(emotion, emotion)
                lines.append(f"  • {emotion_pt}: {percentage}")
        else:
            lines.append("Nenhuma emoção detectada")
        
        lines.append("")
        
        # Atividades Detectadas
        lines.append("-" * 80)
        lines.append("ATIVIDADES DETECTADAS")
        lines.append("-" * 80)
        
        activity_dist = summary['activities']['distribution']
        if activity_dist:
            lines.append(f"Atividade principal: {summary['activities']['dominant']}")
            lines.append("")
            lines.append("Distribuição de atividades:")
            
            # Tradução de atividades
            activity_translations = {
                'standing': 'Em pé',
                'sitting': 'Sentado',
                'arms_up': 'Braços levantados',
                'crouching': 'Agachado',
                'leaning': 'Inclinado',
                'walking': 'Caminhando',
                'waving': 'Acenando',
                'unknown': 'Desconhecido'
            }
            
            for activity, count in activity_dist.items():
                activity_pt = activity_translations.get(activity, activity)
                lines.append(f"  • {activity_pt}: {count} detecções")
        else:
            lines.append("Nenhuma atividade detectada")
        
        lines.append("")
        
        # Anomalias Detectadas
        lines.append("-" * 80)
        lines.append("ANOMALIAS DETECTADAS")
        lines.append("-" * 80)
        lines.append(f"Total de anomalias: {summary['anomalies']['total']}")
        lines.append("")
        
        if summary['anomalies']['by_type']:
            lines.append("Anomalias por tipo:")
            
            # Tradução de tipos de anomalias
            anomaly_translations = {
                'sudden_movement': 'Movimentos bruscos',
                'abnormal_pose': 'Poses anômalas',
                'rapid_emotion_change': 'Mudanças emocionais súbitas',
                'sustained_extreme_emotion': 'Emoções extremas sustentadas'
            }
            
            for anomaly_type, count in summary['anomalies']['by_type'].items():
                anomaly_pt = anomaly_translations.get(anomaly_type, anomaly_type)
                lines.append(f"  • {anomaly_pt}: {count} ocorrências")
            
            lines.append("")
            
            # Listar detalhes das anomalias (primeiras 20)
            if summary['anomalies']['details']:
                lines.append("Detalhes das anomalias (primeiras 20):")
                for i, anomaly in enumerate(summary['anomalies']['details'][:20], 1):
                    timestamp_str = self.format_timestamp(anomaly['timestamp'])
                    lines.append(f"  {i}. [{timestamp_str}] Frame {anomaly['frame']}: {anomaly['description']}")
        else:
            lines.append("Nenhuma anomalia detectada")
        
        lines.append("")
        
        # Timeline
        if self.stats.timeline:
            lines.append("-" * 80)
            lines.append("LINHA DO TEMPO")
            lines.append("-" * 80)
            
            for timestamp, event in self.stats.timeline[:50]:  # Primeiros 50 eventos
                timestamp_str = self.format_timestamp(timestamp)
                lines.append(f"[{timestamp_str}] {event}")
            
            if len(self.stats.timeline) > 50:
                lines.append(f"\n... e mais {len(self.stats.timeline) - 50} eventos")
            
            lines.append("")
        
        # Rodapé
        lines.append("=" * 80)
        lines.append("FIM DO RELATÓRIO")
        lines.append("=" * 80)
        
        # Escrever arquivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Relatório gerado: {output_path}")
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """
        Gera estatísticas resumidas
        
        Returns:
            Dicionário com estatísticas principais
        """
        summary = self.stats.get_summary()
        
        return {
            'total_frames': summary['general']['total_frames'],
            'frames_with_faces': summary['general']['frames_with_faces'],
            'top_emotions': summary['emotions'].get('top_5', []),
            'dominant_activity': summary['activities']['dominant'],
            'total_anomalies': summary['anomalies']['total'],
        }
