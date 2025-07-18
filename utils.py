"""
Módulo com funções utilitárias gerais para pipeline Pseudo-LiDAR
"""

import socket
import time
import torch
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

from config import OBJECT_COLORS, DEFAULT_COLOR, SRS_SERVER_IP, SRS_RTMP_PORT

logger = logging.getLogger(__name__)


def get_device() -> str:
    """
    Detecta automaticamente o melhor dispositivo disponível.
    
    Returns:
        String do dispositivo ('cuda' ou 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def print_device_info(device: str) -> None:
    """
    Exibe informações sobre o dispositivo sendo usado.
    
    Args:
        device: String do dispositivo
    """
    print(f"🚀 Usando dispositivo: {device.upper()}")
    
    if device == 'cuda' and torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔥 Versão CUDA: {torch.version.cuda}")
    elif device == 'cuda':
        print("⚠️ CUDA solicitada mas não disponível. Usando CPU.")
    else:
        print("💻 Usando processamento em CPU")


def get_object_color(class_name: str) -> Tuple[int, int, int]:
    """
    Retorna a cor específica para uma classe de objeto.
    
    Args:
        class_name: Nome da classe do objeto
        
    Returns:
        Tupla BGR da cor do objeto
    """
    return OBJECT_COLORS.get(class_name, DEFAULT_COLOR)


def validate_input_source(source: str) -> bool:
    """
    Valida se a fonte de entrada é acessível.
    
    Args:
        source: URL ou caminho da fonte
        
    Returns:
        True se válida
    """
    try:
        if source.startswith(('rtmp://', 'rtsp://', 'http://', 'https://')):
            # Validação básica de URL
            return True
        else:
            # Validação de arquivo local
            import os
            return os.path.exists(source)
    except Exception as e:
        logger.error(f"Erro validando fonte: {e}")
        return False


def validate_server_config(server_ip: str, port: str) -> bool:
    """
    Valida a conectividade com o servidor SRS.
    
    Args:
        server_ip: IP do servidor
        port: Porta do servidor
        
    Returns:
        True se conectável
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((server_ip, int(port)))
        sock.close()
        return result == 0
    except Exception as e:
        logger.error(f"Erro validando servidor: {e}")
        return False


def test_connection(server_ip: str = SRS_SERVER_IP, port: int = int(SRS_RTMP_PORT)) -> bool:
    """
    Testa conectividade com servidor SRS.
    
    Args:
        server_ip: IP do servidor SRS
        port: Porta do servidor
        
    Returns:
        True se conectável
    """
    return validate_server_config(server_ip, str(port))


def create_stats_dict() -> Dict[str, Any]:
    """
    Cria dicionário base para estatísticas do pipeline.
    
    Returns:
        Dicionário com estatísticas inicializadas
    """
    return {
        'total_frames': 0,
        'processed_frames': 0,
        'bev_generations': 0,
        'depth_estimations': 0,
        'detections': 0,
        'fps': 0.0,
        'processing_time_ms': 0.0,
        'start_time': time.time(),
        'last_fps_time': time.time(),
        'fps_counter': 0,
        'memory_usage_mb': 0.0,
        'gpu_memory_mb': 0.0
    }


def update_fps_stats(stats: Dict[str, Any]) -> None:
    """
    Atualiza estatísticas de FPS.
    
    Args:
        stats: Dicionário de estatísticas
    """
    current_time = time.time()
    stats['fps_counter'] += 1
    
    if current_time - stats['last_fps_time'] >= 1.0:
        stats['fps'] = stats['fps_counter'] / (current_time - stats['last_fps_time'])
        stats['fps_counter'] = 0
        stats['last_fps_time'] = current_time


def format_fps(fps: float) -> str:
    """Formata FPS para exibição."""
    return f"{fps:.1f}"


def format_memory_usage(device: str) -> str:
    """
    Formata informações de uso de memória.
    
    Args:
        device: Dispositivo em uso
        
    Returns:
        String formatada com uso de memória
    """
    if device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        return f"GPU: {allocated:.1f}MB (cache: {cached:.1f}MB)"
    else:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024**2
        return f"RAM: {memory_mb:.1f}MB"


def format_processing_time(time_ms: float) -> str:
    """Formata tempo de processamento."""
    return f"{time_ms:.1f}ms"


def print_final_stats(stats: Dict[str, Any]) -> None:
    """
    Imprime estatísticas finais do pipeline.
    
    Args:
        stats: Dicionário com estatísticas
    """
    total_time = time.time() - stats['start_time']
    avg_fps = stats['processed_frames'] / total_time if total_time > 0 else 0
    
    print("\n" + "="*60)
    print("🎯 ESTATÍSTICAS FINAIS - PSEUDO-LIDAR PIPELINE")
    print("="*60)
    print(f"⏱️  Tempo total: {total_time:.1f}s")
    print(f"🎬 Frames processados: {stats['processed_frames']}")
    print(f"🔍 Estimativas de profundidade: {stats['depth_estimations']}")
    print(f"🗺️  Gerações BEV: {stats['bev_generations']}")
    print(f"📦 Detecções totais: {stats['detections']}")
    print(f"⚡ FPS médio: {avg_fps:.1f}")
    print(f"🚀 Tempo médio por frame: {stats['processing_time_ms']:.1f}ms")
    print(f"💾 Uso de memória: {format_memory_usage('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*60)


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Redimensiona frame mantendo proporção.
    
    Args:
        frame: Frame de entrada
        target_size: Tamanho alvo (width, height)
        
    Returns:
        Frame redimensionado
    """
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_depth(depth_map: np.ndarray, min_depth: float = 0.1, max_depth: float = 100.0) -> np.ndarray:
    """
    Normaliza mapa de profundidade para visualização.
    
    Args:
        depth_map: Mapa de profundidade
        min_depth: Profundidade mínima
        max_depth: Profundidade máxima
        
    Returns:
        Mapa normalizado [0, 255]
    """
    depth_map = np.clip(depth_map, min_depth, max_depth)
    depth_normalized = (depth_map - min_depth) / (max_depth - min_depth)
    return (depth_normalized * 255).astype(np.uint8)


def apply_colormap(depth_map: np.ndarray, colormap: int = cv2.COLORMAP_PLASMA) -> np.ndarray:
    """
    Aplica colormap ao mapa de profundidade.
    
    Args:
        depth_map: Mapa de profundidade normalizado
        colormap: Colormap do OpenCV
        
    Returns:
        Imagem colorida
    """
    return cv2.applyColorMap(depth_map, colormap)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Valida configurações do pipeline.
    
    Args:
        config: Configurações a validar
        
    Returns:
        True se válidas
    """
    required_keys = ['focal_length_x', 'focal_length_y', 'principal_point_x', 'principal_point_y']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Configuração obrigatória ausente: {key}")
            return False
            
    return True 