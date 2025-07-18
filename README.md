# Pseudo-LiDAR Inference Pipeline

Um pipeline completo de inferência **Pseudo-LiDAR** para geração de visualizações **Bird's Eye View (BEV)** em tempo real com streaming integrado, baseado na mesma infraestrutura do projeto YOLO.

## 🎯 Objetivo

Este projeto implementa um sistema de visão computacional que:
- **Estima profundidade** de imagens monoculares usando modelos estado-da-arte (DPT, MiDaS)
- **Gera vista Bird's Eye View** através de transformações geométricas 3D
- **Detecta objetos na BEV** (opcional) usando YOLO
- **Transmite resultados** em tempo real via streaming RTMP para o mesmo servidor do projeto YOLO

## 🚀 Funcionalidades

- **Estimativa de profundidade** usando modelos DPT ou MiDaS
- **Conversão para nuvem de pontos 3D** com projeção de câmera
- **Geração de vista BEV** com grade de referência
- **Detecções opcionais** de objetos na vista BEV usando YOLO
- **Streaming em tempo real** com visualização lado a lado (Original | Profundidade | BEV)
- **Arquitetura modular** adaptada do projeto YOLO
- **Suporte a GPU/CPU** com otimizações de performance
- **Configuração flexível** para diferentes cenários de câmera

## 📁 Estrutura do Projeto

```
pseudo-lidar-infer/
├── config.py                 # Configurações centralizadas
├── pipeline.py               # Classe principal PseudoLiDARPipeline  
├── depth_processor.py        # Processamento de profundidade e BEV
├── streaming.py              # Gerenciamento de streaming FFmpeg
├── pseudo_lidar_utils.py     # Utilitários específicos Pseudo-LiDAR
├── utils.py                  # Utilitários gerais (adaptados do YOLO)
├── __init__.py               # Inicialização do pacote
├── requirements.txt          # Dependências
├── main_notebook.ipynb       # Notebook principal de uso
├── README.md                 # Esta documentação
├── diagrams/                 # Diagramas e documentação visual
└── models/                   # Modelos pré-treinados (download automático)
```

## 🔧 Instalação

### 1. Instalar Dependências

```bash
cd pseudo-lidar-infer
pip install -r requirements.txt
```

### 2. Dependências Principais

- **PyTorch** ≥ 2.0.0 (com suporte CUDA se disponível)
- **OpenCV** ≥ 4.7.0 (processamento de imagem)
- **Transformers** ≥ 4.21.0 (modelos DPT da Hugging Face)
- **MiDaS** ≥ 1.0.0 (modelo de profundidade alternativo)
- **Ultralytics** ≥ 8.0.0 (YOLO para detecções opcionais)
- **FFmpeg** (streaming de vídeo)

### 3. Verificar Instalação

```python
import pseudo_lidar_infer as pli
pli.check_dependencies()
```

## 🎮 Uso Básico

### Execução Rápida

```python
from pseudo_lidar_infer import run_pseudo_lidar_pipeline

# Usar webcam (câmera 0)
success = run_pseudo_lidar_pipeline(source=0, max_frames=1000)

# Processar arquivo de vídeo
success = run_pseudo_lidar_pipeline(source="video.mp4")

# Processar stream RTMP (mesmo do YOLO)
success = run_pseudo_lidar_pipeline(source="rtmp://195.200.0.55:1935/live/livestream")
```

### Uso Avançado

```python
from pseudo_lidar_infer import PseudoLiDARPipeline

# Criar pipeline
pipeline = PseudoLiDARPipeline()

# Carregar modelos
pipeline.load_models(model_name="DPT_Large")

# Configurar streaming para servidor SRS
pipeline.setup_streaming("rtmp://195.200.0.55:1935/live/bev_processed")

# Processar vídeo
pipeline.process_video_source("input.mp4", max_frames=500)

# Obter estatísticas
stats = pipeline.get_stats()
print(f"BEV geradas: {stats['bev_generations']}")
print(f"Detecções: {stats['detections']}")
```

## ⚙️ Configuração

### Configurações de Câmera (`config.py`)

```python
CAMERA_CONFIG = {
    "focal_length_x": 800.0,      # Distância focal X (pixels)
    "focal_length_y": 800.0,      # Distância focal Y (pixels)  
    "principal_point_x": 320.0,   # Centro ótico X
    "principal_point_y": 240.0,   # Centro ótico Y
    "camera_height_meters": 1.6,  # Altura da câmera (metros)
    "camera_tilt_degrees": 0.0,   # Inclinação da câmera
}
```

### Configurações BEV

```python
BEV_CONFIG = {
    "range_meters": 50.0,         # Alcance da vista (metros)
    "resolution_meters": 0.1,     # Resolução por pixel (metros/pixel)
    "output_size": (500, 500),    # Tamanho da imagem BEV
    "center_offset": (250, 400),  # Posição da câmera na BEV
}
```

### Modelos Suportados

- **DPT_Large**: Estado-da-arte para estimativa de profundidade
- **DPT_Hybrid**: Versão otimizada do DPT
- **MiDaS**: Modelo clássico, mais rápido
- **MiDaS_small**: Versão leve do MiDaS

## 🌐 Streaming

O pipeline envia a visualização processada para o mesmo servidor SRS usado pelo projeto YOLO:

- **Servidor**: `195.200.0.55:1935`
- **URL de entrada**: `rtmp://195.200.0.55:1935/live/livestream` (mesmo do YOLO)
- **URL de saída BEV**: `rtmp://195.200.0.55:1935/live/bev_processed` (diferente do YOLO)

### Visualização Final

O stream contém uma imagem lado a lado (1920x480):
1. **Esquerda**: Imagem original (640x480)
2. **Centro**: Mapa de profundidade colorido (640x480)  
3. **Direita**: Vista BEV com detecções opcionais (640x480)

## 🔬 Como Funciona

### 1. Estimativa de Profundidade
- Usa modelos neurais (DPT/MiDaS) para estimar profundidade por pixel
- Entrada: Imagem RGB [H, W, 3]
- Saída: Mapa de profundidade [H, W]

### 2. Projeção 3D
- Converte pixels + profundidade em coordenadas 3D usando parâmetros intrínsecos da câmera
- Aplicação: `(u, v, d) → (x, y, z)`

### 3. Transformação para Solo
- Transforma coordenadas da câmera para referencial do solo
- Considera altura e inclinação da câmera

### 4. Projeção BEV
- Projeta pontos 3D em vista superior (Bird's Eye View)
- Cria imagem 2D representando o espaço ao redor da câmera

### 5. Detecções (Opcional)
- Aplica YOLO na vista BEV para detectar objetos
- Mostra bounding boxes diretamente no espaço 3D

## 📊 Performance

### Configurações Otimizadas
- **FPS alvo**: 15 FPS (priorizando acurácia sobre velocidade)
- **Resolução de entrada**: 640x480 (balanceando qualidade e performance)
- **GPU recomendada**: 4GB+ VRAM para DPT_Large
- **CPU**: Suporte a fallback automático

### Benchmarks Esperados
- **DPT_Large + GPU**: ~10-15 FPS
- **MiDaS + GPU**: ~20-25 FPS  
- **CPU**: ~2-5 FPS (dependendo do hardware)

## 🔗 Integração com Projeto YOLO

Este projeto reutiliza a infraestrutura do projeto YOLO:

### Semelhanças
- **Servidor SRS**: Mesmo servidor de streaming
- **Estrutura de código**: Arquitetura modular similar
- **Configurações**: Padrão de config centralizado
- **Utilitários**: Funções adaptadas do YOLO

### Diferenças
- **Modelo**: Pseudo-LiDAR vs YOLO
- **Output**: Vista BEV vs detecções 2D
- **URL stream**: `/bev_processed` vs `/processed`
- **FPS**: 15 FPS vs 25+ FPS (priorizando acurácia)

## 🚀 Execução no Servidor

```bash
# No servidor 195.200.0.55
cd pseudo-lidar-infer

# Execução básica
python -m pseudo_lidar_infer.pipeline

# Com fonte específica
python -m pseudo_lidar_infer.pipeline rtmp://195.200.0.55:1935/live/livestream 1000

# Usando notebook
jupyter notebook main_notebook.ipynb
```

## 📈 Monitoramento

### Estatísticas em Tempo Real
- Frames processados por segundo (FPS)
- Tempo de processamento por frame
- Número de estimativas de profundidade
- Número de vistas BEV geradas
- Detecções de objetos (se habilitado)
- Uso de memória GPU/CPU

### Logs
- Arquivo: `pseudo_lidar_pipeline.log`
- Nível: Configurável (DEBUG, INFO, WARNING, ERROR)

## 🔧 Solução de Problemas

### Erro de Memória GPU
```python
# Reduzir modelo ou usar CPU
DEPTH_MODEL = "MiDaS_small"  # Modelo menor
USE_GPU = False              # Forçar CPU
```

### Streaming Não Funciona
```python
# Testar conectividade
from pseudo_lidar_infer.streaming import test_streaming_server
if not test_streaming_server():
    print("Servidor SRS não acessível")
```

### Performance Baixa
```python
# Otimizações
FPS = 10                    # Reduzir FPS alvo
INPUT_SIZE = (320, 240)     # Reduzir resolução
ENABLE_YOLO_DETECTION = False # Desabilitar detecções
```

## 📝 TODO / Melhorias Futuras

- [ ] **Suporte a câmeras estéreo** para profundidade mais precisa
- [ ] **Calibração automática** de parâmetros de câmera
- [ ] **Fusão temporal** para suavizar vista BEV
- [ ] **Detecção de obstáculos** específica para BEV
- [ ] **Interface web** para monitoramento em tempo real
- [ ] **Otimização TensorRT** para inference mais rápida
- [ ] **Suporte a múltiplas câmeras** 
- [ ] **Exportação de nuvem de pontos** para análise 3D

## 🤝 Contribuição

Este projeto foi desenvolvido como extensão do sistema YOLO existente. Para melhorias:

1. Mantenha compatibilidade com infraestrutura SRS
2. Siga padrões de código do projeto YOLO
3. Documente mudanças de configuração
4. Teste com diferentes modelos de profundidade

## 📄 Licença

Mesmo esquema de licenciamento do projeto YOLO base.

---

**Projeto**: Pseudo-LiDAR Inference Pipeline  
**Baseado em**: YOLO Inference Pipeline  
**Servidor**: 195.200.0.55 (mesma infraestrutura)  
**Status**: ✅ Desenvolvimento concluído - Pronto para testes 