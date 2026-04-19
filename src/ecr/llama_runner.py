"""Inference Llama 2-7B-Chat pour Table 2 (zero-shot + ECR[Llama 2-Chat]).

Deux variantes sont couvertes par ce module :

1. **Llama 2-7B-Chat zero-shot** (Table 2, ligne "Llama 2-7B-Chat")
   Genere une reponse empathique a partir du dialogue history + item
   recommande par ECR, sans fine-tuning.

2. **ECR[Llama 2-Chat]** (Table 2, derniere ligne)
   Charge le LoRA adapter entraine sur `llama_train.json` (2,459 reviews
   empathiques filtrees) et genere les reponses sur le test set.

Les deux cas utilisent le meme template "chat" Llama 2 decrit par Meta :

    <s>[INST] <<SYS>>
    {system}
    <</SYS>>

    {user} [/INST]

Pour un debit raisonnable sur RTX 5090 (32 GB), on supporte deux backends :
- `backend="hf"` : `transformers.pipeline("text-generation")` en bf16,
  ~5-8 samples/s. Simple, aucune dep externe.
- `backend="vllm"` : via API OpenAI-compatible sur http://localhost:8001.
  ~30-50 samples/s, permet de cohabiter avec d'autres jobs sur le meme GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_SYSTEM_PROMPT = (
    "You are an empathetic movie recommendation assistant. Given the prior "
    "dialogue between the user and a recommender agent, and given a movie that "
    "should be recommended next, write ONE empathetic, informative reply that "
    "(a) acknowledges the user's emotional state, (b) naturally recommends the "
    "movie by name, and (c) briefly motivates the recommendation using the "
    "movie's themes. Do not invent external facts about the movie."
)


@dataclass
class Sample:
    dialogue_id: str
    history: str
    item: str
    ground_truth: Optional[str] = None


def build_chat_prompt(history: str, item: str, system: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Format Llama 2 Chat (balises <s>[INST]<<SYS>>...[/INST]).

    L'utilisation avec `transformers.apply_chat_template(..., model="meta-llama/Llama-2-...")`
    produit exactement la meme chaine, mais on la garde explicite ici pour que
    l'output soit reproductible cote vLLM (ou /v1/chat/completions est
    converti silencieusement par le tokenizer embarque).
    """
    user = (
        f"Dialogue history:\n{history.strip()}\n\n"
        f"Recommended movie: {item.strip()}\n\n"
        "Your reply:"
    )
    return (
        f"<s>[INST] <<SYS>>\n{system.strip()}\n<</SYS>>\n\n{user} [/INST]"
    )


def load_test_samples(jsonl_path: Path, limit: Optional[int] = None) -> List[Sample]:
    """Charge `redial_gen/test_data_dbpedia_emo.jsonl` (ou equivalent).

    Le format ECR varie selon la version du script `process*.py` :
    - parfois JSONL (1 objet par ligne) ;
    - parfois JSON global (liste d'objets).
    On supporte les deux.

    Les cles attendues sont parmi : `context` (list) / `dialog_history` (list)
    / `dialogue` (str), `rec` / `movies` / `recommended` / `item`, `resp` /
    `target` / `ground_truth`.
    """
    path = Path(jsonl_path)
    raw_rows: List[dict] = []
    text = path.read_text().strip()
    if text.startswith("["):
        raw_rows = json.loads(text)
    else:
        for line in text.splitlines():
            if line.strip():
                raw_rows.append(json.loads(line))

    samples: List[Sample] = []
    for idx, row in enumerate(raw_rows):
        context = (
            row.get("context")
            or row.get("dialog_history")
            or row.get("dialogue_history")
            or []
        )
        if isinstance(context, str):
            history = context
        else:
            history = "\n".join(
                f"{'User' if i % 2 == 0 else 'Assistant'}: {turn}"
                for i, turn in enumerate(context)
            )
        rec_list = (
            row.get("rec")
            or row.get("movies")
            or row.get("recommended")
            or ([row["item"]] if "item" in row else [""])
        )
        item = rec_list[0] if isinstance(rec_list, list) and rec_list else (
            rec_list if isinstance(rec_list, str) else ""
        )
        samples.append(
            Sample(
                dialogue_id=str(row.get("dialogue_id", row.get("id", idx))),
                history=history,
                item=str(item),
                ground_truth=row.get("resp") or row.get("target") or row.get("ground_truth"),
            )
        )
        if limit and len(samples) >= limit:
            break
    return samples


def _load_model_with_fa2(model_dir, torch_dtype, device_map, token=None,
                          quantization_config=None, use_flash_attn_2=True):
    """Charge un modele HF en essayant Flash-Attention 2 d'abord, avec fallback sdpa.

    Flash-Attention 2 apporte 2-3x sur l'attention Llama 2 (sans impact qualite).
    Si le package `flash-attn` n'est pas installe ou pas compatible sm_120,
    on retombe sur `sdpa` (PyTorch 2.x Scaled Dot-Product Attention), qui est
    deja 1.3-1.5x plus rapide que l'impl eager.
    """
    from transformers import AutoModelForCausalLM
    kwargs = {"torch_dtype": torch_dtype, "device_map": device_map}
    if token is not None:
        kwargs["token"] = token
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs.pop("torch_dtype", None)
    if use_flash_attn_2:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_dir, attn_implementation="flash_attention_2", **kwargs,
            )
        except (ImportError, ValueError) as exc:
            print(f"[llama] flash_attention_2 indisponible ({exc}); fallback sdpa")
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_dir, attn_implementation="sdpa", **kwargs,
        )
    except (ImportError, ValueError):
        return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)


def generate_hf(
    samples: List[Sample],
    model_dir: str,
    lora_dir: Optional[str] = None,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    use_flash_attn_2: bool = True,
) -> List[Dict]:
    """Inference via transformers (bf16). Support LoRA via peft + Flash-Attn 2."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model_with_fa2(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_flash_attn_2=use_flash_attn_2,
    )
    if lora_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()

    outputs = []
    for s in samples:
        prompt = build_chat_prompt(s.history, s.item)
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append({
            "dialogue_id": s.dialogue_id,
            "item": s.item,
            "response": text.strip(),
            "ground_truth": s.ground_truth,
        })
    return outputs


def generate_vllm(
    samples: List[Sample],
    base_url: str = "http://localhost:8001/v1",
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    api_key: str = "EMPTY",
    max_new_tokens: int = 150,
    temperature: float = 0.7,
) -> List[Dict]:
    """Inference via vLLM (API OpenAI-compatible)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    outputs = []
    for s in samples:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Dialogue history:\n{s.history.strip()}\n\n"
                        f"Recommended movie: {s.item.strip()}\n\nYour reply:"
                    ),
                },
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        outputs.append({
            "dialogue_id": s.dialogue_id,
            "item": s.item,
            "response": resp.choices[0].message.content.strip(),
            "ground_truth": s.ground_truth,
        })
    return outputs


def dump_generations(outputs: List[Dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
