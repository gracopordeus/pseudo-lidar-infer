"""
Utilitários específicos para processamento Pseudo-LiDAR
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def validate_bev_config(bev_config: Dict[str, Any]) -> bool:
    """
    Valida configurações de BEV.
    
    Args:
        bev_config: Configurações BEV
        
    Returns:
        True se válida
    """
    required_keys = ['range_meters', 'resolution_meters', 'output_size', 'center_offset']
    
    for key in required_keys:
        if key not in bev_config:
            logger.error(f"Configuração BEV obrigatória ausente: {key}")
            return False
    
    return True


def apply_camera_projection(depth_map: np.ndarray, camera_config: Dict[str, Any]) -> np.ndarray:
    """
    Aplica projeção de câmera para converter profundidade em coordenadas 3D.
    
    Args:
        depth_map: Mapa de profundidade [H, W]
        camera_config: Configurações da câmera
        
    Returns:
        Pontos 3D [N, 3] onde N = H*W
    """
    h, w = depth_map.shape
    
    # Parâmetros intrínsecos da câmera
    fx = camera_config['focal_length_x']
    fy = camera_config['focal_length_y']
    cx = camera_config['principal_point_x']
    cy = camera_config['principal_point_y']
    
    # Criar grade de coordenadas de pixels
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Conversão para coordenadas de câmera
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    
    # Empilhar coordenadas 3D
    points_3d = np.stack([x, y, z], axis=-1)
    
    # Reshape para [N, 3]
    points_3d = points_3d.reshape(-1, 3)
    
    # Filtrar pontos válidos (profundidade > 0)
    valid_mask = points_3d[:, 2] > 0
    points_3d = points_3d[valid_mask]
    
    return points_3d


def transform_to_ground_plane(points_3d: np.ndarray, camera_config: Dict[str, Any]) -> np.ndarray:
    """
    Transforma pontos 3D para o plano do solo.
    
    Args:
        points_3d: Pontos 3D em coordenadas de câmera [N, 3]
        camera_config: Configurações da câmera
        
    Returns:
        Pontos transformados [N, 3]
    """
    # Altura da câmera
    camera_height = camera_config.get('camera_height_meters', 1.6)
    
    # Inclinação da câmera (em radianos)
    tilt_radians = np.radians(camera_config.get('camera_tilt_degrees', 0.0))
    
    # Matriz de rotação para inclinação
    cos_tilt = np.cos(tilt_radians)
    sin_tilt = np.sin(tilt_radians)
    
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_tilt, -sin_tilt],
        [0, sin_tilt, cos_tilt]
    ])
    
    # Aplicar rotação
    points_rotated = points_3d @ rotation_matrix.T
    
    # Translação para ajustar altura da câmera
    translation = np.array([0, 0, -camera_height])
    points_transformed = points_rotated + translation
    
    return points_transformed


def create_bev_from_depth(depth_map: np.ndarray, 
                         camera_config: Dict[str, Any], 
                         bev_config: Dict[str, Any]) -> np.ndarray:
    """
    Cria vista Bird's Eye View a partir de mapa de profundidade.
    
    Args:
        depth_map: Mapa de profundidade [H, W]
        camera_config: Configurações da câmera
        bev_config: Configurações da vista BEV
        
    Returns:
        Imagem BEV [H_bev, W_bev]
    """
    try:
        # 1. Projeção 3D
        points_3d = apply_camera_projection(depth_map, camera_config)
        
        if len(points_3d) == 0:
            logger.warning("Nenhum ponto 3D válido encontrado")
            return np.zeros(bev_config['output_size'][::-1], dtype=np.uint8)
        
        # 2. Transformação para plano do solo
        points_ground = transform_to_ground_plane(points_3d, camera_config)
        
        # 3. Projeção BEV
        bev_image = project_to_bev(points_ground, bev_config)
        
        return bev_image
        
    except Exception as e:
        logger.error(f"Erro na criação da BEV: {e}")
        return np.zeros(bev_config['output_size'][::-1], dtype=np.uint8)


def project_to_bev(points_3d: np.ndarray, bev_config: Dict[str, Any]) -> np.ndarray:
    """
    Projeta pontos 3D na vista Bird's Eye View.
    
    Args:
        points_3d: Pontos 3D [N, 3]
        bev_config: Configurações BEV
        
    Returns:
        Imagem BEV [H, W]
    """
    range_meters = bev_config['range_meters']
    resolution = bev_config['resolution_meters']
    output_size = bev_config['output_size']  # (width, height)
    center_offset = bev_config['center_offset']  # (x_offset, y_offset)
    
    # Filtrar pontos dentro do alcance
    valid_mask = (
        (np.abs(points_3d[:, 0]) < range_meters) &
        (points_3d[:, 2] < range_meters) &
        (points_3d[:, 2] > 0)
    )
    valid_points = points_3d[valid_mask]
    
    if len(valid_points) == 0:
        return np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    
    # Conversão para coordenadas de pixel BEV
    pixel_x = (valid_points[:, 0] / resolution + center_offset[0]).astype(int)
    pixel_y = (center_offset[1] - valid_points[:, 2] / resolution).astype(int)
    
    # Filtrar pixels dentro da imagem
    valid_pixels = (
        (pixel_x >= 0) & (pixel_x < output_size[0]) &
        (pixel_y >= 0) & (pixel_y < output_size[1])
    )
    
    pixel_x = pixel_x[valid_pixels]
    pixel_y = pixel_y[valid_pixels]
    
    # Criar imagem BEV
    bev_image = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    
    if len(pixel_x) > 0:
        # Calcular intensidade baseada na altura (Y)
        heights = valid_points[valid_pixels, 1]
        
        # Normalizar alturas para intensidade [0, 255]
        height_min, height_max = -2.0, 2.0  # Assumindo alturas entre -2m e +2m
        intensity = np.clip((heights - height_min) / (height_max - height_min) * 255, 0, 255).astype(np.uint8)
        
        # Atribuir intensidades aos pixels
        bev_image[pixel_y, pixel_x] = intensity
    
    return bev_image


def enhance_bev_image(bev_image: np.ndarray) -> np.ndarray:
    """
    Melhora a imagem BEV com filtros e processamento.
    
    Args:
        bev_image: Imagem BEV original [H, W]
        
    Returns:
        Imagem BEV melhorada [H, W, 3] (colorida)
    """
    # Aplicar filtro gaussiano para suavizar
    smoothed = cv2.GaussianBlur(bev_image, (3, 3), 0.5)
    
    # Aplicar dilatação para preencher gaps pequenos
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(smoothed, kernel, iterations=1)
    
    # Aplicar colormap para visualização
    colored = cv2.applyColorMap(dilated, cv2.COLORMAP_VIRIDIS)
    
    return colored


def add_grid_to_bev(bev_image: np.ndarray, bev_config: Dict[str, Any]) -> np.ndarray:
    """
    Adiciona grade de referência à imagem BEV.
    
    Args:
        bev_image: Imagem BEV [H, W, 3]
        bev_config: Configurações BEV
        
    Returns:
        Imagem BEV com grade [H, W, 3]
    """
    result = bev_image.copy()
    h, w = result.shape[:2]
    
    resolution = bev_config['resolution_meters']
    grid_spacing_meters = 5.0  # Grid a cada 5 metros
    grid_spacing_pixels = int(grid_spacing_meters / resolution)
    
    # Cor da grade (cinza claro)
    grid_color = (100, 100, 100)
    
    # Linhas verticais
    for x in range(0, w, grid_spacing_pixels):
        cv2.line(result, (x, 0), (x, h), grid_color, 1)
    
    # Linhas horizontais
    for y in range(0, h, grid_spacing_pixels):
        cv2.line(result, (0, y), (w, y), grid_color, 1)
    
    # Adicionar marcador central (posição da câmera)
    center_x, center_y = bev_config['center_offset']
    if 0 <= center_x < w and 0 <= center_y < h:
        cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)  # Vermelho
        cv2.circle(result, (center_x, center_y), 8, (255, 255, 255), 2)  # Branco
    
    return result


def filter_depth_outliers(depth_map: np.ndarray, 
                         min_depth: float = 0.1, 
                         max_depth: float = 100.0) -> np.ndarray:
    """
    Filtra outliers no mapa de profundidade.
    
    Args:
        depth_map: Mapa de profundidade [H, W]
        min_depth: Profundidade mínima válida
        max_depth: Profundidade máxima válida
        
    Returns:
        Mapa de profundidade filtrado
    """
    # Filtrar valores fora do intervalo
    filtered = np.clip(depth_map, min_depth, max_depth)
    
    # Aplicar filtro mediano para remover ruído
    filtered = cv2.medianBlur(filtered.astype(np.float32), 3)
    
    return filtered


def create_depth_colormap(depth_map: np.ndarray, 
                         min_depth: float = 0.1, 
                         max_depth: float = 50.0) -> np.ndarray:
    """
    Cria visualização colorida do mapa de profundidade.
    
    Args:
        depth_map: Mapa de profundidade [H, W]
        min_depth: Profundidade mínima para normalização
        max_depth: Profundidade máxima para normalização
        
    Returns:
        Imagem colorida [H, W, 3]
    """
    # Normalizar profundidade
    depth_normalized = np.clip((depth_map - min_depth) / (max_depth - min_depth), 0, 1)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Aplicar colormap
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    
    return colored 