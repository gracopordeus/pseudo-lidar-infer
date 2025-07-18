#!/usr/bin/env python3
"""
Script de teste para verificar imports do pipeline Pseudo-LiDAR
"""

import sys
import os

# Adicionar diretório atual ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_imports():
    """Testa todos os imports necessários."""
    
    print("🧪 Testando imports do pipeline Pseudo-LiDAR...")
    
    # Teste 1: Config
    try:
        from config import (
            CAMERA_CONFIG, BEV_CONFIG, DEPTH_MODEL, 
            SRS_SERVER_IP, OUTPUT_RTMP_URL, INPUT_SOURCE
        )
        print("✅ Config importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro no import de config: {e}")
        return False
    
    # Teste 2: Utils
    try:
        from utils import get_device, print_device_info, test_connection
        print("✅ Utils importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro no import de utils: {e}")
        return False
    
    # Teste 3: Streaming  
    try:
        from streaming import test_streaming_server, FFmpegStreamer
        print("✅ Streaming importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro no import de streaming: {e}")
        return False
    
    # Teste 4: Pseudo LiDAR Utils
    try:
        from pseudo_lidar_utils import create_bev_from_depth
        print("✅ Pseudo LiDAR Utils importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro no import de pseudo_lidar_utils: {e}")
        return False
    
    # Teste 5: Depth Processor
    try:
        from depth_processor import DepthProcessor
        print("✅ Depth Processor importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro no import de depth_processor: {e}")
        return False
    
    # Teste 6: Pipeline Principal
    try:
        from pipeline import PseudoLiDARPipeline, create_pipeline_from_config
        print("✅ Pipeline Principal importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro no import de pipeline: {e}")
        return False
    
    print("\n🎉 Todos os imports foram bem-sucedidos!")
    
    # Teste básico de funcionalidade
    try:
        device = get_device()
        print(f"🖥️ Dispositivo detectado: {device}")
        
        pipeline = create_pipeline_from_config()
        print("✅ Pipeline criado com sucesso")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Erro ao testar funcionalidade básica: {e}")
        return False

def test_dependencies():
    """Testa dependências externas."""
    
    print("\n🔍 Testando dependências externas...")
    
    # PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🚀 CUDA disponível: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 CUDA não disponível, usando CPU")
    except ImportError:
        print("❌ PyTorch não instalado")
        return False
    
    # OpenCV
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV não instalado")
        return False
    
    # Transformers
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers não instalado")
        return False
    
    # YOLO (opcional)
    try:
        from ultralytics import YOLO
        print("✅ YOLO (Ultralytics) disponível")
    except ImportError:
        print("⚠️ YOLO não disponível (opcional)")
    
    return True

if __name__ == "__main__":
    print(f"📁 Diretório de trabalho: {os.getcwd()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    success = True
    
    success &= test_dependencies()
    success &= test_imports()
    
    if success:
        print("\n🎉 Todos os testes passaram! Pipeline pronto para uso.")
        sys.exit(0)
    else:
        print("\n❌ Alguns testes falharam. Verifique as dependências e arquivos.")
        sys.exit(1) 