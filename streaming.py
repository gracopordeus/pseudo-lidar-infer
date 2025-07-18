"""
Módulo para streaming de vídeo BEV com FFmpeg
Adaptado do projeto YOLO para saídas Pseudo-LiDAR
"""

import subprocess
import threading
import time
from typing import Optional, List, Tuple
import numpy as np
import logging

from config import STREAMING_CONFIG, SRS_SERVER_IP, SRS_RTMP_PORT
from utils import test_connection

logger = logging.getLogger(__name__)


class FFmpegStreamer:
    """
    Classe para gerenciar streaming de vídeo BEV com FFmpeg.
    """
    
    def __init__(self, output_url: str, width: int = 1920, height: int = 480, fps: int = 15):
        """
        Inicializa o streamer FFmpeg para BEV.
        
        Args:
            output_url: URL de saída do stream
            width: Largura do vídeo (lado a lado: 3x640 = 1920)
            height: Altura do vídeo
            fps: Taxa de frames por segundo (reduzida para BEV)
        """
        self.output_url = output_url
        self.width = width
        self.height = height
        self.fps = fps
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self._error_count = 0
        self._max_errors = 5
        
        logger.info(f"Streamer BEV inicializado: {width}x{height}@{fps}fps -> {output_url}")
        
    def start(self) -> bool:
        """
        Inicia o processo de streaming.
        
        Returns:
            True se iniciado com sucesso
        """
        try:
            cmd = self._build_ffmpeg_command()
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            self.is_running = True
            self._error_count = 0
            
            # Thread para monitorar erros
            self._start_error_monitor()
            
            logger.info("✅ Streaming BEV iniciado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro iniciando streaming: {e}")
            return False
    
    def _build_ffmpeg_command(self) -> List[str]:
        """
        Constrói comando FFmpeg otimizado para BEV.
        
        Returns:
            Lista com comando FFmpeg
        """
        config = STREAMING_CONFIG
        
        cmd = [
            'ffmpeg',
            '-y',  # Sobrescrever arquivo de saída
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # Input do stdin
            
            # Configurações de encoding
            '-c:v', config['codec'],
            '-preset', config['preset'],
            '-b:v', config['bitrate'],
            '-maxrate', config['maxrate'],
            '-bufsize', config['buffer_size'],
            '-pix_fmt', config['pixel_format'],
            
            # Otimizações para streaming de BEV
            '-g', str(self.fps * 2),  # Keyframe a cada 2 segundos
            '-keyint_min', str(self.fps),
            '-sc_threshold', '0',
            
            # Filtros de vídeo para BEV
            '-vf', self._build_video_filters(),
            
            # Configurações de streaming
            '-f', 'flv',
            self.output_url
        ]
        
        logger.debug(f"Comando FFmpeg: {' '.join(cmd)}")
        return cmd
    
    def _build_video_filters(self) -> str:
        """
        Constrói filtros de vídeo específicos para BEV.
        
        Returns:
            String com filtros FFmpeg
        """
        filters = [
            # Equalização para melhor visualização
            'eq=contrast=1.1:brightness=0.05',
            
            # Opcional: marca d'água
            f"drawtext=text='Pseudo-LiDAR BEV':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5"
        ]
        
        return ','.join(filters)
    
    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Envia um frame para o stream.
        
        Args:
            frame: Frame BEV processado [H, W, 3]
            
        Returns:
            True se enviado com sucesso
        """
        if not self.is_running or self.process is None:
            return False
        
        try:
            # Verificar dimensões
            if frame.shape[:2] != (self.height, self.width):
                logger.warning(f"Redimensionando frame de {frame.shape[:2]} para ({self.height}, {self.width})")
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Converter para bytes
            frame_bytes = frame.tobytes()
            
            # Enviar para FFmpeg
            self.process.stdin.write(frame_bytes)
            self.process.stdin.flush()
            
            return True
            
        except BrokenPipeError:
            logger.error("Pipe quebrado - processo FFmpeg encerrado")
            self.is_running = False
            return False
        except Exception as e:
            self._error_count += 1
            logger.error(f"Erro enviando frame: {e}")
            
            if self._error_count >= self._max_errors:
                logger.error("Muitos erros - parando streaming")
                self.stop()
            
            return False
    
    def _start_error_monitor(self):
        """Inicia thread para monitorar erros do FFmpeg."""
        def monitor():
            if self.process:
                for line in iter(self.process.stderr.readline, b''):
                    line_str = line.decode('utf-8').strip()
                    if line_str:
                        if 'error' in line_str.lower() or 'failed' in line_str.lower():
                            logger.error(f"FFmpeg: {line_str}")
                        elif logger.level <= logging.DEBUG:
                            logger.debug(f"FFmpeg: {line_str}")
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def stop(self):
        """Para o streaming."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Forçando término do processo FFmpeg")
                self.process.kill()
            except Exception as e:
                logger.error(f"Erro parando streaming: {e}")
        
        self.is_running = False
        logger.info("🛑 Streaming BEV parado")
    
    def is_healthy(self) -> bool:
        """
        Verifica se o streaming está funcionando corretamente.
        
        Returns:
            True se saudável
        """
        if not self.is_running or self.process is None:
            return False
        
        return self.process.poll() is None and self._error_count < self._max_errors


def test_streaming_server(server_ip: str = SRS_SERVER_IP, 
                         port: int = int(SRS_RTMP_PORT)) -> bool:
    """
    Testa conectividade com servidor de streaming.
    
    Args:
        server_ip: IP do servidor SRS
        port: Porta do servidor
        
    Returns:
        True se conectável
    """
    return test_connection(server_ip, port)


def create_bev_streamer(output_url: str, 
                       width: int = 1920, 
                       height: int = 480, 
                       fps: int = 15) -> FFmpegStreamer:
    """
    Cria streamer configurado para visualizações BEV.
    
    Args:
        output_url: URL de saída RTMP
        width: Largura (para lado a lado: 3x640)
        height: Altura padrão
        fps: FPS (reduzido para BEV)
        
    Returns:
        Instância configurada do FFmpegStreamer
    """
    streamer = FFmpegStreamer(output_url, width, height, fps)
    
    logger.info(f"Streamer BEV criado: {width}x{height}@{fps}fps")
    return streamer


def validate_streaming_config() -> bool:
    """
    Valida configurações de streaming.
    
    Returns:
        True se configurações válidas
    """
    config = STREAMING_CONFIG
    
    required_keys = ['bitrate', 'preset', 'pixel_format', 'codec']
    for key in required_keys:
        if key not in config:
            logger.error(f"Configuração de streaming ausente: {key}")
            return False
    
    # Validar valores
    valid_presets = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
    if config['preset'] not in valid_presets:
        logger.warning(f"Preset possivelmente inválido: {config['preset']}")
    
    return True


class StreamingMonitor:
    """
    Monitor para múltiplos streams BEV.
    """
    
    def __init__(self):
        self.streamers: List[FFmpegStreamer] = []
        self.stats = {
            'frames_sent': 0,
            'errors': 0,
            'start_time': time.time()
        }
    
    def add_streamer(self, streamer: FFmpegStreamer):
        """Adiciona streamer ao monitor."""
        self.streamers.append(streamer)
        logger.info(f"Streamer adicionado ao monitor: {streamer.output_url}")
    
    def send_frame_to_all(self, frame: np.ndarray) -> int:
        """
        Envia frame para todos os streamers.
        
        Args:
            frame: Frame a enviar
            
        Returns:
            Número de streamers que receberam com sucesso
        """
        success_count = 0
        
        for streamer in self.streamers:
            if streamer.is_healthy():
                if streamer.send_frame(frame):
                    success_count += 1
                else:
                    self.stats['errors'] += 1
        
        if success_count > 0:
            self.stats['frames_sent'] += 1
        
        return success_count
    
    def get_healthy_streamers(self) -> List[FFmpegStreamer]:
        """Retorna lista de streamers saudáveis."""
        return [s for s in self.streamers if s.is_healthy()]
    
    def stop_all(self):
        """Para todos os streamers."""
        for streamer in self.streamers:
            streamer.stop()
        
        logger.info(f"Todos os {len(self.streamers)} streamers parados")
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do monitor."""
        uptime = time.time() - self.stats['start_time']
        healthy_count = len(self.get_healthy_streamers())
        
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'active_streamers': healthy_count,
            'total_streamers': len(self.streamers),
            'avg_fps': self.stats['frames_sent'] / uptime if uptime > 0 else 0
        } 