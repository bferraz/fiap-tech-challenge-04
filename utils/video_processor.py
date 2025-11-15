"""
Utilitários para processamento de vídeo
"""
import cv2
from typing import Optional, Tuple


class VideoProcessor:
    """Classe para gerenciar a captura e gravação de vídeos"""
    
    def __init__(self, video_path: str):
        """
        Inicializa o processador de vídeo
        
        Args:
            video_path: Caminho para o arquivo de vídeo
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Erro ao abrir o vídeo: {video_path}")
        
        # Propriedades do vídeo
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        self.current_frame = 0
    
    def read_frame(self) -> Tuple[bool, Optional[any]]:
        """
        Lê o próximo frame do vídeo
        
        Returns:
            Tuple contendo (sucesso, frame)
        """
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame
    
    def get_timestamp(self) -> float:
        """
        Retorna o timestamp atual em segundos
        
        Returns:
            Timestamp em segundos
        """
        return self.current_frame / self.fps if self.fps > 0 else 0
    
    def get_progress_percentage(self) -> float:
        """
        Retorna o progresso de processamento em porcentagem
        
        Returns:
            Porcentagem de progresso (0-100)
        """
        return (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0
    
    def release(self):
        """Libera os recursos do vídeo"""
        self.cap.release()
    
    @staticmethod
    def create_video_writer(output_path: str, fps: float, width: int, height: int, codec: str = 'mp4v'):
        """
        Cria um objeto VideoWriter para salvar vídeos
        
        Args:
            output_path: Caminho de saída do vídeo
            fps: Frames por segundo
            width: Largura do vídeo
            height: Altura do vídeo
            codec: Codec de vídeo (padrão: mp4v)
        
        Returns:
            Objeto cv2.VideoWriter
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Formata segundos para formato HH:MM:SS
        
        Args:
            seconds: Tempo em segundos
        
        Returns:
            String formatada (HH:MM:SS)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
