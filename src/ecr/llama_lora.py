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


# ---------------------------------------------------------------------------
# Compat shim transformers 4.57 <-> accelerate < 1.3
# ---------------------------------------------------------------------------
#
# A partir de transformers ~4.50, `Trainer._wrap_model` appelle
# `self.accelerator.unwrap_model(model, keep_torch_compile=False)`. Le kwarg
# `keep_torch_compile` a ete AJOUTE dans accelerate 1.3.0 (nov 2024). Sur
# accelerate < 1.3 (notamment 1.1.1 pinne par notre constraint NGC actuel),
# le kwarg leve TypeError.
#
# Cette fonction detecte la situation au runtime et wrappe `unwrap_model` pour
# absorber le kwarg inconnu. Une fois accelerate >= 1.3 installe (future build
# Docker qui aura la contrainte relachee), le shim se transforme en no-op.
# Idempotent : safe a appeler plusieurs fois.
def _ensure_accelerator_unwrap_compat() -> None:
    try:
        import inspect as _inspect
        from accelerate import Accelerator as _Accelerator
    except ImportError:
        return
    if getattr(_Accelerator.unwrap_model, "_ecr_shimmed", False):
        return  # deja patche
    try:
        sig = _inspect.signature(_Accelerator.unwrap_model)
    except (TypeError, ValueError):
        return
    if "keep_torch_compile" in sig.parameters:
        return  # accelerate >= 1.3, rien a faire

    _original_unwrap = _Accelerator.unwrap_model

    def _patched_unwrap(self, model, keep_fp32_wrapper=True,
                         keep_torch_compile=None, **kwargs):
        # `keep_torch_compile` est ignore (comportement accelerate < 1.3 :
        # le `torch.compile` wrapper etait deja unwrap par defaut).
        return _original_unwrap(self, model,
                                 keep_fp32_wrapper=keep_fp32_wrapper, **kwargs)

    _patched_unwrap._ecr_shimmed = True  # marker anti-double-patch
    _Accelerator.unwrap_model = _patched_unwrap
    print("[lora] shim installe : Accelerator.unwrap_model accepte "
          "keep_torch_compile= pour compat transformers 4.57 (accelerate < 1.3)")


# ---------------------------------------------------------------------------
# Template prompt pour ECR[Llama 2-Chat]
# ---------------------------------------------------------------------------
#
# Le format REEL du dataset `llama_train.json` fourni par les auteurs suit une
# structure Alpaca-style `{instruction, input, output}` (pas `{context,
# emotion, item, review}` comme on aurait pu le penser). Exemple d'une entree :
#
#   {
#     "instruction": "You are a recommender chatting with the user to provide
#                     movie recommendation. Please continue generating based on
#                     the First Sentence. Please utilize the information about
#                     Movie Name, Related Entities, and Related Knowledge from
#                     KG.",
#     "input": "Movie Name: The Man Who Knew Infinity (2015)\n"
#              "Related Entities: Full moon, Amazing Stories, Education, ...\n"
#              "Related Knowledge from KG: None\n"
#              "First Sentence: I haven't seen that one. I have ...",
#     "output": "Anyone who rates this movie under the 7 is insensible ..."
#   }
#
# Le modele est entraine a CONTINUER la "First Sentence" presente dans le
# champ `input`, en s'appuyant sur le nom de film + entites + knowledge KG.
# C'est un task de "review continuation" empathique, pas un task de chat
# multi-tour classique. Pour l'inference, on utilise `llama_test.json` (format
# identique, output vide), on construit le meme prompt, et on decode la
# continuation.
PROMPT_TEMPLATE = (
    "<s>[INST] <<SYS>>\n"
    "{instruction}\n"
    "<</SYS>>\n\n"
    "{input} [/INST] "
)


def _load_llama_json(path: Path) -> List[Dict]:
    """Charge `llama_train.json` / `llama_test.json` (format Alpaca ECR).

    Format attendu :
        [{"instruction": "...", "input": "...", "output": "..."}, ...]
    """
    with Path(path).open() as f:
        data = json.load(f)
    return data


def build_lora_prompt(instruction: str, input_text: str) -> str:
    """Construit un prompt LoRA identique a celui du training.

    Utilise par `train_lora` (via `build_training_examples`) et par l'inference
    `generate_hf_lora` dans `llama_runner.py`. Garder cette fonction comme
    SOURCE UNIQUE de verite pour le template -- tout mismatch train/infer
    donne des sorties polluees de placeholders comme `[Insert the user's
    emotion]` (observe en pratique avec l'ancien template).
    """
    return PROMPT_TEMPLATE.format(
        instruction=str(instruction).strip(),
        input=str(input_text).strip(),
    )


def build_training_examples(path: Path) -> List[Dict[str, str]]:
    """Format {"prompt": ..., "target": ...} pour `datasets.Dataset.from_list`."""
    raw = _load_llama_json(path)
    examples = []
    for row in raw:
        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        output_text = str(row.get("output", "")).strip()
        if not instruction or not input_text or not output_text:
            # Ligne invalide -> skip (evite d'empoisonner le LoRA avec des
            # exemples vides, ce qui etait le bug du template precedent).
            continue
        examples.append({
            "prompt": build_lora_prompt(instruction, input_text),
            "target": output_text + " </s>",
        })
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
    # Compat shim obligatoire AVANT d'importer Trainer : corrige TypeError sur
    # `Accelerator.unwrap_model(..., keep_torch_compile=False)` que transformers
    # 4.57 emet. No-op si accelerate >= 1.3.
    _ensure_accelerator_unwrap_compat()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
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
        # `prepare_model_for_kbit_training` active gradient_checkpointing en
        # interne (avec use_reentrant=True par defaut, warning benin sur torch
        # 2.10+ mais fatal sur 2.9). On pourrait passer
        # gradient_checkpointing_kwargs={"use_reentrant": False} ici mais la
        # signature varie selon la version de peft -- on laisse le defaut.
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

    # Gradient checkpointing aussi en path bf16 : SANS ca, Llama-2-7B bf16
    # + batch 4 + seq 1024 sature les 32 GB d'une 5090 (activations non
    # checkpointees = ~10-12 GB). En path 4-bit, prepare_model_for_kbit_training
    # l'a deja active. On evite le double-enable.
    if not use_4bit:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # requis pour PEFT + gradient_checkpointing : les embeddings d'entree
        # doivent exposer require_grad pour que la chaine de gradients traverse
        # les couches checkpointees vers les adapters LoRA.
        model.enable_input_require_grads()

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

    collator = DataCollatorForSeq2Seq(tokenizer)

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
        # Note : en 4-bit le checkpointing est DEJA actif via
        # `prepare_model_for_kbit_training`. En bf16 il est active manuellement
        # ci-dessus (`model.gradient_checkpointing_enable(...)`). On garde
        # `gradient_checkpointing=True` ici + `use_reentrant=False` pour que
        # le Trainer de transformers detecte l'etat et ne re-active pas
        # redondamment.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    try:
        trainer.train()
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return output_dir
    finally:
        # Nettoyage VRAM agressif : training LoRA sur Llama-2-7B bf16 peak
        # autour de 18-25 GB (poids + activations + gradients + optimizer).
        # Sans `del` explicite, le trainer + le model + l'optimizer + les
        # dataloaders restent reachables pendant toute la vie du kernel
        # Jupyter, bloquant plusieurs GB qu'on ne pourrait pas ceder au
        # scorer Qwen en phase 4. `gc.collect` + `empty_cache` ne recuperent
        # rien tant qu'il y a des references Python vivantes.
        # NB : `del locals()[name]` ne fonctionne PAS en scope fonction
        # (locals() est un snapshot), d'ou les dels explicites ci-dessous
        # avec try/except pour le cas ou un nom a ete shadowe/reassigne.
        import gc
        try: del trainer
        except Exception: pass
        try: del model
        except Exception: pass
        try: del tokenizer
        except Exception: pass
        try: del collator
        except Exception: pass
        try: del ds
        except Exception: pass
        try: del examples
        except Exception: pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
