"""
Pipeline principal para processamento Pseudo-LiDAR e geração de vista BEV
"""

import cv2
import time
import numpy as np
from typing import Optional, Dict, Any
import logging

from config import (
    DEPTH_MODEL, CAMERA_CONFIG, BEV_CONFIG, CONFIDENCE_THRESHOLD, 
    INPUT_SIZE, FPS, OUTPUT_RTMP_URL, ENABLE_YOLO_DETECTION, YOLO_MODEL
)
from depth_processor import DepthProcessor
from streaming import FFmpegStreamer, test_streaming_server, create_bev_streamer
from utils import (
    get_device, print_device_info, validate_input_source,
    validate_server_config, create_stats_dict, print_final_stats
)

logger = logging.getLogger(__name__)


class PseudoLiDARPipeline:
    """
    Pipeline principal para processamento Pseudo-LiDAR com geração BEV e streaming.
    """
    
    def __init__(self, camera_config: Dict[str, Any] = None, bev_config: Dict[str, Any] = None):
        """
        Inicializa o pipeline Pseudo-LiDAR.
        
        Args:
            camera_config: Configurações da câmera (usa padrão se None)
            bev_config: Configurações da vista BEV (usa padrão se None)
        """
        self.camera_config = camera_config or CAMERA_CONFIG
        self.bev_config = bev_config or BEV_CONFIG
        self.device = get_device()
        
        # Componentes principais
        self.depth_processor: Optional[DepthProcessor] = None
        self.streamer: Optional[FFmpegStreamer] = None
        
        # Estatísticas
        self.stats = create_stats_dict()
        
        print("🚀 Pipeline Pseudo-LiDAR inicializado")
        print_device_info(self.device)
        print(f"📷 Modelo de profundidade: {DEPTH_MODEL}")
        print(f"🗺️  Configuração BEV: {self.bev_config['output_size']} @ {self.bev_config['range_meters']}m")
    
    def load_models(self, model_name: str = DEPTH_MODEL) -> bool:
        """
        Carrega os modelos de profundidade e YOLO.
        
        Args:
            model_name: Nome do modelo de profundidade
            
        Returns:
            True se carregado com sucesso
        """
        try:
            print(f"📥 Carregando modelo de profundidade: {model_name}")
            
            # Inicializar processador de profundidade
            self.depth_processor = DepthProcessor(
                self.camera_config, 
                self.bev_config, 
                model_name, 
                self.device
            )
            
            # Carregar modelos
            success = self.depth_processor.load_models(
                enable_yolo=ENABLE_YOLO_DETECTION,
                yolo_model=YOLO_MODEL
            )
            
            if success:
                print("✅ Modelos carregados com sucesso")
                return True
            else:
                print("❌ Erro carregando modelos")
                return False
                
        except Exception as e:
            logger.error(f"Erro carregando modelos: {e}")
            print(f"❌ Erro carregando modelos: {e}")
            return False
    
    def setup_streaming(self, output_url: str = OUTPUT_RTMP_URL) -> bool:
        """
        Configura streaming de vídeo BEV.
        
        Args:
            output_url: URL de saída do stream
            
        Returns:
            True se configurado com sucesso
        """
        try:
            print(f"🌐 Configurando streaming: {output_url}")
            
            # Testar conectividade
            if not test_streaming_server():
                print("⚠️ Aviso: Servidor de streaming não acessível")
                # Continua mesmo assim para testes locais
            
            # Criar streamer para visualização lado a lado (3 x 640 = 1920 pixels)
            self.streamer = create_bev_streamer(
                output_url=output_url,
                width=1920,  # 3 vistas lado a lado
                height=480,  # Altura padrão
                fps=FPS
            )
            
            # Iniciar streaming
            if self.streamer.start():
                print("✅ Streaming iniciado com sucesso")
                return True
            else:
                print("❌ Erro iniciando streaming")
                return False
                
        except Exception as e:
            logger.error(f"Erro configurando streaming: {e}")
            print(f"❌ Erro configurando streaming: {e}")
            return False
    
    def process_video_source(self, source: str, max_frames: int = None) -> bool:
        """
        Processa fonte de vídeo (webcam, arquivo, stream).
        
        Args:
            source: Fonte de vídeo (0 para webcam, caminho arquivo, URL stream)
            max_frames: Máximo de frames a processar (None = ilimitado)
            
        Returns:
            True se processado com sucesso
        """
        if self.depth_processor is None:
            print("❌ Modelos não carregados. Execute load_models() primeiro.")
            return False
        
        # Validar fonte
        if not validate_input_source(str(source)):
            print(f"❌ Fonte de vídeo inválida: {source}")
            return False
        
        print(f"🎬 Iniciando processamento: {source}")
        
        # Abrir captura de vídeo
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Não foi possível abrir a fonte: {source}")
            return False
        
        try:
            frame_count = 0
            start_time = time.time()
            
            print("🔄 Processando frames... (Pressione 'q' para parar)")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("📺 Fim do vídeo ou erro na captura")
                    break
                
                # Processar frame
                success = self._process_single_frame(frame, frame_count)
                
                if success:
                    frame_count += 1
                    self.stats['processed_frames'] += 1
                
                self.stats['total_frames'] += 1
                
                # Atualizar estatísticas a cada 30 frames
                if frame_count % 30 == 0:
                    self._print_progress_stats(frame_count, start_time)
                
                # Verificar limite de frames
                if max_frames and frame_count >= max_frames:
                    print(f"🎯 Limite de {max_frames} frames atingido")
                    break
                
                # Verificar se deve parar (implementação futura para interface)
                # if self._should_stop():
                #     break
            
            # Estatísticas finais
            total_time = time.time() - start_time
            print(f"\n✅ Processamento concluído!")
            print(f"📊 {frame_count} frames processados em {total_time:.1f}s")
            print(f"⚡ FPS médio: {frame_count / total_time:.1f}")
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ Processamento interrompido pelo usuário")
            return False
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            print(f"❌ Erro no processamento: {e}")
            return False
        finally:
            cap.release()
            if self.streamer:
                self.streamer.stop()
    
    def _process_single_frame(self, frame: np.ndarray, frame_number: int) -> bool:
        """
        Processa um único frame.
        
        Args:
            frame: Frame de entrada
            frame_number: Número do frame
            
        Returns:
            True se processado com sucesso
        """
        try:
            # Processar frame com Pseudo-LiDAR
            results = self.depth_processor.process_frame(
                frame, enable_detection=ENABLE_YOLO_DETECTION
            )
            
            if not results['success']:
                logger.warning(f"Falha no processamento do frame {frame_number}")
                return False
            
            # Criar visualização lado a lado
            visualization = self.depth_processor.create_side_by_side_visualization(results)
            
            # Enviar para streaming se disponível
            if self.streamer and self.streamer.is_healthy():
                self.streamer.send_frame(visualization)
            
            # Atualizar estatísticas
            self._update_stats(results)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro processando frame {frame_number}: {e}")
            return False
    
    def _update_stats(self, results: Dict[str, Any]):
        """Atualiza estatísticas do pipeline."""
        self.stats['depth_estimations'] += 1
        self.stats['bev_generations'] += 1
        self.stats['detections'] += results.get('detections', 0)
        self.stats['processing_time_ms'] = results.get('processing_time_ms', 0)
        
        # Atualizar FPS
        from utils import update_fps_stats
        update_fps_stats(self.stats)
    
    def _print_progress_stats(self, frame_count: int, start_time: float):
        """Imprime estatísticas de progresso."""
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"📊 Frame {frame_count:6d} | "
              f"FPS: {fps:5.1f} | "
              f"Detecções: {self.stats['detections']:4d} | "
              f"Tempo: {self.stats['processing_time_ms']:5.1f}ms | "
              f"Mem: {self._get_memory_usage()}")
    
    def _get_memory_usage(self) -> str:
        """Retorna uso de memória formatado."""
        from utils import format_memory_usage
        return format_memory_usage(self.device)
    
    def run_from_config(self, source: str = None, max_frames: int = None) -> bool:
        """
        Executa pipeline completo usando configurações padrão.
        
        Args:
            source: Fonte de vídeo (usa padrão se None)
            max_frames: Máximo de frames (None = ilimitado)
            
        Returns:
            True se executado com sucesso
        """
        from config import INPUT_SOURCE
        
        source = source or INPUT_SOURCE
        
        print("🎬 Iniciando pipeline Pseudo-LiDAR...")
        
        # 1. Carregar modelos
        if not self.load_models():
            return False
        
        # 2. Configurar streaming  
        if not self.setup_streaming():
            print("⚠️ Streaming não disponível, continuando sem streaming...")
        
        # 3. Processar vídeo
        success = self.process_video_source(source, max_frames)
        
        # 4. Imprimir estatísticas finais
        if success:
            print_final_stats(self.stats)
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do pipeline."""
        pipeline_stats = self.stats.copy()
        
        if self.depth_processor:
            processor_stats = self.depth_processor.get_stats()
            pipeline_stats.update(processor_stats)
        
        return pipeline_stats
    
    def cleanup(self):
        """Limpa recursos do pipeline."""
        if self.streamer:
            self.streamer.stop()
        
        # Limpeza de memória GPU
        if self.device == 'cuda':
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        
        print("🧹 Pipeline limpo")


def create_pipeline_from_config() -> PseudoLiDARPipeline:
    """
    Cria pipeline usando configurações padrão.
    
    Returns:
        Instância do pipeline configurada
    """
    return PseudoLiDARPipeline(CAMERA_CONFIG, BEV_CONFIG)


# Função de conveniência para execução rápida
def run_pseudo_lidar_pipeline(source: str = None, max_frames: int = None) -> bool:
    """
    Executa pipeline Pseudo-LiDAR com configurações padrão.
    
    Args:
        source: Fonte de vídeo (webcam=0, arquivo, URL)
        max_frames: Máximo de frames a processar
        
    Returns:
        True se executado com sucesso
    """
    pipeline = create_pipeline_from_config()
    return pipeline.run_from_config(source, max_frames)


if __name__ == "__main__":
    # Execução direta do módulo
    import sys
    
    source = sys.argv[1] if len(sys.argv) > 1 else None
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    success = run_pseudo_lidar_pipeline(source, max_frames)
    sys.exit(0 if success else 1) 
