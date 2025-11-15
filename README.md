# Analisador de Vídeo - Tech Challenge Fase 4

Sistema completo de análise de vídeo com reconhecimento facial, análise de emoções, detecção de atividades e identificação de anomalias.

## 📋 Requisitos

- Python 3.11 (recomendado)
- Windows/Linux/macOS
- Webcam (opcional, para testes)

## 🚀 Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/bferraz/fiap-tech-challenge-04.git
cd fiap-tech-challenge-04
```

### 2. Criar ambiente virtual Python 3.11

```powershell
# Windows PowerShell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalar dependências

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Nota:** A instalação pode demorar alguns minutos devido ao TensorFlow e outras bibliotecas pesadas.

## 📁 Estrutura do Projeto

```
.
├── main.py                 # Script principal de execução
├── requirements.txt        # Dependências do projeto
├── video.mp4              # Vídeo de entrada (você deve adicionar)
├── config/
│   └── settings.py        # Configurações gerais
├── src/
│   ├── video_analyzer.py  # Orquestrador principal
│   ├── face_detector.py   # Detecção de rostos com IDs
│   ├── emotion_analyzer.py # Análise de emoções
│   ├── activity_detector.py # Detecção de atividades
│   ├── anomaly_detector.py  # Detecção de anomalias
│   └── report_generator.py  # Geração de relatórios
├── utils/
│   ├── video_processor.py      # Processamento de vídeo
│   └── statistics_collector.py # Coleta de estatísticas
└── output/
    ├── video_processado.mp4    # Vídeo com anotações
    ├── relatorio.txt           # Relatório em texto
    └── relatorio.json          # Relatório em JSON
```

## 🎯 Como Usar

### 1. Preparar o vídeo

Coloque seu vídeo com o nome `video.mp4` no diretório raiz do projeto.

### 2. Executar a análise

```powershell
python main.py
```

### 3. Resultados

Após o processamento, você encontrará na pasta `output/`:

- **video_processado.mp4**: Vídeo com todas as anotações visuais
- **relatorio.txt**: Relatório completo em texto
- **relatorio.json**: Dados estruturados em JSON

## 🎨 Funcionalidades

### ✅ Reconhecimento Facial
- Detecção automática de rostos
- Rastreamento entre frames
- Atribuição de IDs únicos (rosto_1, rosto_2, etc.)

### ✅ Análise de Emoções
- Detecção de 7 emoções: feliz, triste, raiva, surpresa, medo, desgosto, neutro
- Análise frame a frame
- Estatísticas de emoções por pessoa
- Emoção dominante no vídeo

### ✅ Detecção de Atividades
- Em pé
- Sentado
- Braços levantados
- Agachado
- Inclinado
- Caminhando
- Acenando

### ✅ Detecção de Anomalias
- Movimentos bruscos
- Poses anômalas
- Mudanças emocionais súbitas
- Emoções extremas sustentadas

### ✅ Relatório Automático
- Total de frames analisados
- Número de rostos identificados
- Distribuição de emoções
- Distribuição de atividades
- Número de anomalias detectadas
- Timeline de eventos

## ⚙️ Configurações

Edite `config/settings.py` para ajustar:

- Caminhos de entrada/saída
- Thresholds de detecção
- Parâmetros de visualização
- Sensibilidade de anomalias

## 🧪 Testando com o Vídeo Fornecido

1. Baixe o vídeo fornecido na plataforma do aluno
2. Renomeie para `video.mp4`
3. Coloque no diretório raiz
4. Execute `python main.py`

## 📊 Exemplo de Relatório

```
================================================================================
RELATÓRIO DE ANÁLISE DE VÍDEO
================================================================================

INFORMAÇÕES GERAIS
--------------------------------------------------------------------------------
Total de frames analisados: 1500
Rostos únicos identificados: 2
Taxa de detecção: 95.6%

ANÁLISE DE EMOÇÕES
--------------------------------------------------------------------------------
Emoção dominante: happy
Distribuição de emoções:
  • Feliz: 65.3%
  • Neutro: 20.1%
  • Surpresa: 14.6%

ATIVIDADES DETECTADAS
--------------------------------------------------------------------------------
Atividade principal: standing
Distribuição de atividades:
  • Em pé: 800 detecções
  • Acenando: 300 detecções
  • Caminhando: 200 detecções

ANOMALIAS DETECTADAS: 5
--------------------------------------------------------------------------------
Anomalias por tipo:
  • Movimentos bruscos: 3 ocorrências
  • Mudanças emocionais súbitas: 2 ocorrências
```

## 🛠️ Solução de Problemas

### Erro: "Import could not be resolved"
Isso é apenas um aviso do linter. As bibliotecas serão instaladas com `pip install -r requirements.txt`.

### Erro ao instalar dlib
No Windows, pode ser necessário instalar o Visual C++ Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Vídeo processado está vazio
Verifique se o codec está correto. Tente alterar `VIDEO_CONFIG['codec']` em `config/settings.py` para `'XVID'` ou `'H264'`.

### Processamento muito lento
- Reduza a resolução do vídeo
- Aumente `process_every_n_frames` em `config/settings.py` para processar menos frames
- Use GPU se disponível

## 📝 Observações

- O processamento pode ser lento dependendo do tamanho do vídeo e do hardware
- Para vídeos longos (>5 minutos), considere processar em partes
- Use Python 3.11 para melhor compatibilidade
- TensorFlow pode mostrar avisos - isso é normal

## 👥 Autores

Bruno Ferraz - RM359670

## 📄 Licença

Este projeto faz parte do Tech Challenge - FIAP Pós Tech
