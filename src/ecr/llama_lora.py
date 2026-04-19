"""Fine-tuning LoRA de Llama 2-7B-Chat pour ECR[Llama 2-Chat].

Article section 4.3 / 5.4 :
- corpus : `llama_train.json` (2,459 reviews IMDb filtrees) ;
- 15 epochs, lr=5e-5, effective batch = 16 ;
- prompt : dialogue history + emotion tag + recommended item -> review empathique.

Section 4.3 specifie un fine-tuning instruction-style : le prompt contient le
dialogue + l'emotion-aligned instruction, et la target est la review. On
calcule la loss uniquement sur les tokens de la target (loss masking).

Default LoRA : r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj],
dropout=0.05 -- matche les recettes modernes (Llama 2-Chat + Alpaca/LIMA).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


PROMPT_TEMPLATE = (
    "<s>[INST] <<SYS>>\n"
    "You are an empathetic movie recommendation assistant. Respond to the user "
    "with emotional awareness using the provided emotion tag, and recommend the "
    "given movie with a brief motivation rooted in the movie's themes.\n"
    "<</SYS>>\n\n"
    "Dialogue history:\n{history}\n\n"
    "User emotion: {emotion}\n"
    "Recommended movie: {item}\n\n"
    "Write the empathetic reply: [/INST] "
)


def _load_llama_json(path: Path) -> List[Dict]:
    """Charge `llama_train.json` / `llama_test.json` de l'archive `emo_data.zip`.

    Format attendu (depuis l'archive ECR) :
        [
            {"context": [...], "emotion": "happy", "item": "Inception (2010)",
             "review": "... empathetic review text ..."},
            ...
        ]
    """
    with Path(path).open() as f:
        data = json.load(f)
    return data


def build_training_examples(path: Path) -> List[Dict[str, str]]:
    """Format {"text": prompt+target} pour `datasets.Dataset.from_list`."""
    raw = _load_llama_json(path)
    examples = []
    for row in raw:
        history = "\n".join(row.get("context", []))
        prompt = PROMPT_TEMPLATE.format(
            history=history,
            emotion=row.get("emotion", "neutral"),
            item=row.get("item", ""),
        )
        target = str(row.get("review", "")).strip() + " </s>"
        examples.append({"prompt": prompt, "target": target})
    return examples


def train_lora(
    base_model: str,
    train_json: Path,
    output_dir: Path,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    epochs: int = 15,
    lr: float = 5e-5,
    per_device_batch: int = 4,
    grad_accum: int = 4,
    max_len: int = 1024,
    hf_token: Optional[str] = None,
    use_4bit: bool = False,
    use_flash_attn_2: bool = True,
):
    """Entraine un LoRA adapter et le sauve dans `output_dir`.

    Retourne le chemin de l'adapter sauve (peut etre charge par peft).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "token": hf_token}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs.pop("torch_dtype", None)

    # Flash-Attention 2 : 2-3x sur l'attention Llama 2, fallback automatique sdpa.
    if use_flash_attn_2:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, attn_implementation="flash_attention_2", **model_kwargs,
            )
        except (ImportError, ValueError) as exc:
            print(f"[lora] flash_attention_2 indisponible ({exc}); fallback sdpa")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model, attn_implementation="sdpa", **model_kwargs,
                )
            except (ImportError, ValueError):
                model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    examples = build_training_examples(train_json)
    ds = Dataset.from_list(examples)

    def tokenize_fn(example):
        prompt_ids = tokenizer(
            example["prompt"], truncation=True, max_length=max_len // 2,
            add_special_tokens=False,
        )["input_ids"]
        target_ids = tokenizer(
            example["target"], truncation=True, max_length=max_len // 2,
            add_special_tokens=False,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids[:]
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=not use_4bit,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
