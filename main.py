"""
Script principal para executar a análise de vídeo
"""
import os
import sys

# Adicionar diretório do projeto ao path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.video_analyzer import VideoAnalyzer


def main():
    """Função principal"""
    print("=" * 80)
    print("ANALISADOR DE VÍDEO - TECH CHALLENGE FASE 4")
    print("=" * 80)
    print()
    
    # Verificar se o vídeo existe
    video_path = "video.mp4"
    
    if not os.path.exists(video_path):
        print(f"ERRO: Vídeo não encontrado: {video_path}")
        print(f"Por favor, coloque o arquivo 'video.mp4' no diretório raiz do projeto.")
        print(f"Diretório atual: {os.getcwd()}")
        return
    
    # Criar diretório de saída se não existir
    os.makedirs("output", exist_ok=True)
    
    # Criar analisador e processar vídeo
    try:
        analyzer = VideoAnalyzer(video_path)
        analyzer.process_video()
    except KeyboardInterrupt:
        print("\n\nProcessamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\nERRO durante o processamento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
