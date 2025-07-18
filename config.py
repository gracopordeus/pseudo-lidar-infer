"""
Configurações centralizadas para o pipeline Pseudo-LiDAR
"""

# =============================================================================
# CONFIGURAÇÕES DO SERVIDOR SRS - USANDO MESMA INFRAESTRUTURA DO YOLO
# =============================================================================

# Configurações do servidor SRS
SRS_SERVER_IP = "195.200.0.55"  # IP da VPS (mesmo do projeto YOLO)
SRS_RTMP_PORT = "1935"
SRS_SRT_PORT = "10080"

# URLs de entrada e saída
INPUT_SOURCE = "rtmp://195.200.0.55:1935/live/livestream"  # Mesmo input do YOLO
OUTPUT_RTMP_URL = f"rtmp://{SRS_SERVER_IP}:{SRS_RTMP_PORT}/live/bev_processed"  # Diferente do YOLO
OUTPUT_SRT_URL = f"srt://{SRS_SERVER_IP}:{SRS_SRT_PORT}/bev"

# =============================================================================
# CONFIGURAÇÕES DO MODELO PSEUDO-LIDAR
# =============================================================================

# Configurações de profundidade
DEPTH_MODEL = "DPT_Large"  # Opções: MiDaS_small, MiDaS, DPT_Large, DPT_Hybrid
DEPTH_MODEL_TYPE = "dpt_large"  # Para carregamento do modelo
USE_GPU = True

# Configurações de câmera (necessárias para projeção 3D)
CAMERA_CONFIG = {
    "focal_length_x": 800.0,      # Distância focal em pixels (eixo X)
    "focal_length_y": 800.0,      # Distância focal em pixels (eixo Y)
    "principal_point_x": 320.0,   # Ponto principal X (cx)
    "principal_point_y": 240.0,   # Ponto principal Y (cy)
    "camera_height_meters": 1.6,  # Altura da câmera em metros
    "camera_tilt_degrees": 0.0,   # Inclinação da câmera em graus
}

# Configurações de BEV (Bird's Eye View)
BEV_CONFIG = {
    "range_meters": 50.0,         # Alcance da vista BEV em metros
    "resolution_meters": 0.1,     # Resolução por pixel em metros
    "output_size": (500, 500),    # Tamanho da imagem BEV final
    "center_offset": (250, 400),  # Offset do centro da câmera na imagem BEV
}

# =============================================================================
# CONFIGURAÇÕES DE PROCESSAMENTO
# =============================================================================

# Configurações de vídeo
INPUT_SIZE = (640, 480)
FPS = 15  # FPS reduzido para priorizar acurácia sobre velocidade
CONFIDENCE_THRESHOLD = 0.5

# Configurações de streaming
STREAMING_CONFIG = {
    "bitrate": "2000k",
    "preset": "medium",  # Qualidade média para BEV
    "pixel_format": "yuv420p",
    "codec": "libx264",
    "buffer_size": "4000k",
    "maxrate": "2500k"
}

# =============================================================================
# CONFIGURAÇÕES DE DETECÇÃO (OPCIONAL - INTEGRAÇÃO COM YOLO)
# =============================================================================

ENABLE_YOLO_DETECTION = True  # Se True, aplica YOLO na vista BEV
YOLO_MODEL = "yolov8n.pt"     # Modelo YOLO para detecções na BEV
YOLO_CONFIDENCE = 0.5

# Classes de objetos para detecção na BEV
RELEVANT_CLASSES = [
    "person", "car", "bicycle", "motorcycle", "bus", "truck"
]

# Cores para visualização (BGR format)
OBJECT_COLORS = {
    "person": (0, 255, 0),      # Verde
    "car": (255, 0, 0),         # Azul
    "bicycle": (0, 255, 255),   # Amarelo
    "motorcycle": (255, 0, 255), # Magenta
    "bus": (128, 0, 128),       # Roxo
    "truck": (0, 128, 255),     # Laranja
}

DEFAULT_COLOR = (255, 255, 255)  # Branco

# =============================================================================
# CONFIGURAÇÕES DE LOGGING E DEBUG
# =============================================================================

SAVE_DEBUG_IMAGES = False
DEBUG_OUTPUT_DIR = "debug_output"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Configurações de performance
MAX_PROCESSING_TIME_MS = 100  # Tempo máximo por frame em ms
ENABLE_PERFORMANCE_MONITORING = True 