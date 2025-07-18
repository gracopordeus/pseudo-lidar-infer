"""
Pseudo-LiDAR Inference Pipeline

Um pipeline completo de inferência Pseudo-LiDAR para geração de visualizações
Bird's Eye View (BEV) em tempo real com streaming integrado.

Autor: Sistema Automatizado
Versão: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Sistema Automatizado"
__description__ = "Pipeline de inferência Pseudo-LiDAR com BEV e streaming em tempo real"

# Imports principais
from config import (
    DEPTH_MODEL, BEV_CONFIG, CAMERA_CONFIG, 
    INPUT_SOURCE, OUTPUT_RTMP_URL,
    ENABLE_YOLO_DETECTION, YOLO_MODEL
)

from pipeline import PseudoLiDARPipeline
from depth_processor import DepthProcessor
from streaming import FFmpegStreamer
from utils import get_device, print_device_info, validate_input_source
from pseudo_lidar_utils import (
    create_bev_from_depth, 
    apply_camera_projection,
    validate_bev_config
)

# Configuração de logging
import logging

def setup_logging(level: str = "INFO"):
    """
    Configura o logging para o pipeline.
    
    Args:
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pseudo_lidar_pipeline.log')
        ]
    )

# Constantes públicas
__all__ = [
    # Classes principais
    'PseudoLiDARPipeline',
    'DepthProcessor', 
    'FFmpegStreamer',
    
    # Configurações
    'DEPTH_MODEL',
    'BEV_CONFIG',
    'CAMERA_CONFIG',
    'INPUT_SOURCE',
    'OUTPUT_RTMP_URL',
    
    # Utilitários
    'get_device',
    'print_device_info',
    'validate_input_source',
    'create_bev_from_depth',
    'apply_camera_projection',
    'validate_bev_config',
    'setup_logging',
    
    # Metadados
    '__version__',
    '__author__',
    '__description__'
]

# Aviso sobre dependências pesadas
def check_dependencies():
    """Verifica se as dependências principais estão instaladas."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
        
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    if missing:
        print(f"⚠️ Dependências faltando: {', '.join(missing)}")
        print("Execute: pip install -r requirements.txt")
        return False
    
    return True

# Verificação automática na importação
if not check_dependencies():
    print("❌ Algumas dependências não foram encontradas.")
    print("🔧 Certifique-se de instalar todas as dependências antes de usar o pipeline.") 