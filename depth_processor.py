"""
Módulo para processamento de profundidade e geração de vista BEV
"""

import cv2
import numpy as np
import torch
import time
from typing import Dict, Any, Tuple, Optional
import logging

# Imports condicionais para modelos de profundidade
try:
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    DPT_AVAILABLE = True
except ImportError:
    DPT_AVAILABLE = False
    logging.warning("Transformers não disponível. DPT não será usado.")

# MiDaS via torch.hub (método correto)
try:
    # Testar se conseguimos carregar via torch.hub
    torch.hub.list('intel-isl/MiDaS', trust_repo=True)
    MIDAS_AVAILABLE = True
    logging.info("MiDaS disponível via torch.hub")
except Exception:
    MIDAS_AVAILABLE = False
    logging.warning("MiDaS não disponível.")

# YOLO para detecções opcionais
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO não disponível para detecções na BEV.")

from pseudo_lidar_utils import (
    create_bev_from_depth, enhance_bev_image, add_grid_to_bev,
    filter_depth_outliers, create_depth_colormap
)
from utils import (
    update_fps_stats, format_memory_usage, get_object_color,
    resize_frame, normalize_depth, apply_colormap
)

logger = logging.getLogger(__name__)


class DepthProcessor:
    """
    Classe principal para processamento de profundidade e geração de vista BEV.
    """
    
    def __init__(self, camera_config: Dict[str, Any], bev_config: Dict[str, Any], 
                 model_name: str = "DPT_Large", device: str = "cuda"):
        """
        Inicializa o processador de profundidade.
        
        Args:
            camera_config: Configurações da câmera
            bev_config: Configurações da vista BEV
            model_name: Nome do modelo de profundidade
            device: Dispositivo de processamento
        """
        self.camera_config = camera_config
        self.bev_config = bev_config
        self.model_name = model_name
        self.device = device
        
        # Modelos
        self.depth_model = None
        self.depth_processor = None
        self.yolo_model = None
        
        # Estatísticas
        self.stats = {
            'depth_estimations': 0,
            'bev_generations': 0,
            'detections': 0,
            'processing_time_ms': 0.0,
            'last_process_time': time.time()
        }
        
        # Cache para otimização
        self._input_size = None
        self._transform_fn = None
        
        logger.info(f"Processador de profundidade inicializado: {model_name} no {device}")
    
    def load_models(self, enable_yolo: bool = True, yolo_model: str = "yolov8n.pt") -> bool:
        """
        Carrega os modelos de profundidade e YOLO.
        
        Args:
            enable_yolo: Se deve carregar YOLO
            yolo_model: Modelo YOLO a carregar
            
        Returns:
            True se carregado com sucesso
        """
        try:
            # Carregar modelo de profundidade
            success = self._load_depth_model()
            if not success:
                return False
            
            # Carregar YOLO se solicitado
            if enable_yolo and YOLO_AVAILABLE:
                self._load_yolo_model(yolo_model)
            
            logger.info("Modelos carregados com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro carregando modelos: {e}")
            return False
    
    def _load_depth_model(self) -> bool:
        """Carrega modelo de profundidade."""
        try:
            if self.model_name.startswith("DPT") and DPT_AVAILABLE:
                return self._load_dpt_model()
            elif self.model_name.startswith("MiDaS") and MIDAS_AVAILABLE:
                return self._load_midas_model()
            else:
                logger.error(f"Modelo não suportado ou dependências faltando: {self.model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Erro carregando modelo de profundidade: {e}")
            return False
    
    def _load_dpt_model(self) -> bool:
        """Carrega modelo DPT."""
        model_name = "Intel/dpt-large"
        
        self.depth_processor = DPTImageProcessor.from_pretrained(model_name)
        self.depth_model = DPTForDepthEstimation.from_pretrained(model_name)
        self.depth_model.to(self.device)
        self.depth_model.eval()
        
        logger.info(f"Modelo DPT carregado: {model_name}")
        return True
    
    def _load_midas_model(self) -> bool:
        """Carrega modelo MiDaS via torch.hub."""
        try:
            # Determinar modelo baseado no nome configurado
            if "small" in self.model_name.lower():
                model_type = "MiDaS_small"
            else:
                model_type = "MiDaS"
            
            # Carregar modelo via torch.hub
            self.depth_model = torch.hub.load('intel-isl/MiDaS', model_type, pretrained=True, trust_repo=True)
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
            # Carregar transforms
            transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
            if "small" in model_type.lower():
                self._transform_fn = transforms.small_transform
            else:
                self._transform_fn = transforms.default_transform
            
            logger.info(f"Modelo MiDaS carregado via torch.hub: {model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Erro carregando MiDaS via torch.hub: {e}")
            return False
    
    def _load_yolo_model(self, model_name: str) -> bool:
        """Carrega modelo YOLO."""
        try:
            self.yolo_model = YOLO(model_name)
            self.yolo_model.to(self.device)
            logger.info(f"Modelo YOLO carregado: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Erro carregando YOLO: {e}")
            return False
    
    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estima profundidade de um frame.
        
        Args:
            frame: Frame de entrada [H, W, 3]
            
        Returns:
            Mapa de profundidade [H, W] ou None se erro
        """
        if self.depth_model is None:
            logger.error("Modelo de profundidade não carregado")
            return None
        
        try:
            start_time = time.time()
            
            if self.model_name.startswith("DPT"):
                depth_map = self._estimate_depth_dpt(frame)
            elif self.model_name.startswith("MiDaS"):
                depth_map = self._estimate_depth_midas(frame)
            else:
                return None
            
            # Atualizar estatísticas
            process_time = (time.time() - start_time) * 1000
            self.stats['depth_estimations'] += 1
            self.stats['processing_time_ms'] = process_time
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Erro na estimativa de profundidade: {e}")
            return None
    
    def _estimate_depth_dpt(self, frame: np.ndarray) -> np.ndarray:
        """Estimativa de profundidade usando DPT."""
        # Pré-processar imagem
        inputs = self.depth_processor(images=frame, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inferência
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Pós-processar
        depth_map = predicted_depth.squeeze().cpu().numpy()
        
        # Redimensionar para tamanho original
        depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        
        return depth_map
    
    def _estimate_depth_midas(self, frame: np.ndarray) -> np.ndarray:
        """Estimativa de profundidade usando MiDaS via torch.hub."""
        # Converter BGR para RGB se necessário
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pré-processar usando transform do MiDaS
        input_tensor = self._transform_fn(frame).to(self.device)
        
        # Inferência
        with torch.no_grad():
            prediction = self.depth_model(input_tensor)
            
            # Redimensionar para tamanho original
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        return depth_map
    
    def generate_bev(self, depth_map: np.ndarray) -> Optional[np.ndarray]:
        """
        Gera vista Bird's Eye View a partir do mapa de profundidade.
        
        Args:
            depth_map: Mapa de profundidade [H, W]
            
        Returns:
            Imagem BEV [H, W, 3] ou None se erro
        """
        try:
            # Filtrar outliers
            filtered_depth = filter_depth_outliers(depth_map)
            
            # Criar BEV
            bev_image = create_bev_from_depth(
                filtered_depth, self.camera_config, self.bev_config
            )
            
            # Melhorar visualização
            enhanced_bev = enhance_bev_image(bev_image)
            
            # Adicionar grade
            bev_with_grid = add_grid_to_bev(enhanced_bev, self.bev_config)
            
            self.stats['bev_generations'] += 1
            return bev_with_grid
            
        except Exception as e:
            logger.error(f"Erro na geração BEV: {e}")
            return None
    
    def detect_objects_in_bev(self, bev_image: np.ndarray, 
                             confidence: float = 0.5) -> Tuple[np.ndarray, int]:
        """
        Detecta objetos na vista BEV usando YOLO.
        
        Args:
            bev_image: Imagem BEV [H, W, 3]
            confidence: Limiar de confiança
            
        Returns:
            Tupla (imagem_com_deteccoes, numero_deteccoes)
        """
        if self.yolo_model is None:
            return bev_image, 0
        
        try:
            # Executar detecção
            results = self.yolo_model(bev_image, conf=confidence, verbose=False)
            
            result_image = bev_image.copy()
            total_detections = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extrair informações da detecção
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Nome da classe
                        class_name = self.yolo_model.names[cls]
                        color = get_object_color(class_name)
                        
                        # Desenhar bounding box
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Texto da detecção
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        
                        # Fundo do texto
                        cv2.rectangle(result_image, 
                                    (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1), 
                                    color, -1)
                        
                        # Texto
                        cv2.putText(result_image, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        total_detections += 1
            
            self.stats['detections'] += total_detections
            return result_image, total_detections
            
        except Exception as e:
            logger.error(f"Erro na detecção de objetos: {e}")
            return bev_image, 0
    
    def process_frame(self, frame: np.ndarray, 
                     enable_detection: bool = True) -> Dict[str, Any]:
        """
        Processa um frame completo: profundidade -> BEV -> detecções.
        
        Args:
            frame: Frame de entrada [H, W, 3]
            enable_detection: Se deve executar detecções
            
        Returns:
            Dicionário com resultados do processamento
        """
        start_time = time.time()
        
        # Redimensionar frame se necessário
        if frame.shape[:2] != (480, 640):  # target (height, width)
            frame = resize_frame(frame, (640, 480))  # cv2 espera (width, height)
        
        results = {
            'original_frame': frame,
            'depth_map': None,
            'bev_image': None,
            'processed_bev': None,
            'detections': 0,
            'processing_time_ms': 0.0,
            'success': False
        }
        
        try:
            # 1. Estimativa de profundidade
            depth_map = self.estimate_depth(frame)
            if depth_map is None:
                return results
            
            results['depth_map'] = depth_map
            
            # 2. Geração BEV
            bev_image = self.generate_bev(depth_map)
            if bev_image is None:
                return results
            
            results['bev_image'] = bev_image
            
            # 3. Detecções opcionais
            if enable_detection:
                processed_bev, num_detections = self.detect_objects_in_bev(bev_image)
                results['processed_bev'] = processed_bev
                results['detections'] = num_detections
            else:
                results['processed_bev'] = bev_image
                results['detections'] = 0
            
            # Estatísticas
            processing_time = (time.time() - start_time) * 1000
            results['processing_time_ms'] = processing_time
            results['success'] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no processamento do frame: {e}")
            results['processing_time_ms'] = (time.time() - start_time) * 1000
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do processador."""
        return self.stats.copy()
    
    def create_side_by_side_visualization(self, results: Dict[str, Any]) -> np.ndarray:
        """
        Cria visualização lado a lado: original | profundidade | BEV
        
        Args:
            results: Resultados do processamento
            
        Returns:
            Imagem combinada [H, W*3, 3]
        """
        if not results['success']:
            # Imagem de erro
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "ERRO NO PROCESSAMENTO", (100, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return np.hstack([error_img, error_img, error_img])
        
        # Imagem original
        original = results['original_frame']
        
        # Mapa de profundidade colorido
        depth_colored = create_depth_colormap(results['depth_map'])
        depth_colored = cv2.resize(depth_colored, (640, 480))
        
        # BEV processada
        bev = results['processed_bev']
        bev_resized = cv2.resize(bev, (640, 480))
        
        # Combinar imagens
        combined = np.hstack([original, depth_colored, bev_resized])
        
        # Adicionar labels
        cv2.putText(combined, "ORIGINAL", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "PROFUNDIDADE", (650, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "BEV", (1290, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined 
