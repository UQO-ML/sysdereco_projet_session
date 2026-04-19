"""Helpers pour l'experience ECR (Table 1/3 sweep, Llama-2 LoRA, Qwen scorer).

Modules :
    - metrics_parser  : extraction des metriques depuis les logs loguru/Accelerate
    - llama_runner    : generation Llama 2-7B-Chat (zero-shot + ECR LoRA) via vLLM
    - llama_lora      : fine-tuning LoRA de Llama 2-7B-Chat sur llama_train.json
    - scorer          : client vLLM + prompt Appendix E pour Qwen2.5-32B-AWQ
"""
