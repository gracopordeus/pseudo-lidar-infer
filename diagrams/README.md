# 📊 Diagramas e Documentação Visual

Esta pasta contém diagramas e documentação visual do pipeline Pseudo-LiDAR.

## 🗂️ Conteúdo

### 1. Diagramas de Arquitetura
- **pipeline_architecture.md**: Arquitetura geral do sistema
- **data_flow.md**: Fluxo de dados desde entrada até streaming
- **integration_diagram.md**: Integração com projeto YOLO

### 2. Diagramas de Performance
- **performance_comparison.md**: Comparação com projeto YOLO
- **benchmarks.md**: Benchmarks esperados por modelo
- **optimization_guide.md**: Guia de otimização visual

### 3. Diagramas Técnicos
- **bev_transformation.md**: Processo de transformação BEV
- **depth_estimation.md**: Pipeline de estimativa de profundidade
- **camera_projection.md**: Matemática de projeção de câmera

## 🎯 Como Usar

Cada arquivo `.md` contém:
- Diagramas em formato Mermaid
- Explicações detalhadas
- Exemplos práticos
- Referencias técnicas

## 🔧 Gerando Diagramas

Para gerar diagramas visuais:

```bash
# Instalar mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Gerar imagem PNG
mmdc -i pipeline_architecture.md -o pipeline_architecture.png

# Gerar SVG
mmdc -i data_flow.md -o data_flow.svg
```

## 📚 Referências

- [Mermaid Documentation](https://mermaid-js.github.io/)
- [Projeto YOLO Base](../../yolo-infer/diagrams/)
- [DPT Paper](https://arxiv.org/abs/2103.13413)
- [MiDaS Paper](https://arxiv.org/abs/1907.01341) 