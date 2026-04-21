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
    # Champs specifiques au task LoRA (format Alpaca {instruction, input, output}
    # de `llama_train.json`). Laisses a None pour le path zero-shot (qui
    # construit son prompt via `build_chat_prompt`).
    instruction: Optional[str] = None
    raw_input: Optional[str] = None


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


def build_prompt_for_sample(sample: "Sample") -> str:
    """Dispatch automatique zero-shot vs LoRA en fonction des champs presents.

    - Si `sample.instruction` et `sample.raw_input` sont definis -> format LoRA
      (identique a celui du training llama_lora.py, evite le mismatch
      train/infer qui produisait des placeholders non resolus en sortie).
    - Sinon -> format zero-shot `build_chat_prompt`.
    """
    if sample.instruction and sample.raw_input:
        # Import local pour eviter une dep circulaire.
        from .llama_lora import build_lora_prompt
        return build_lora_prompt(sample.instruction, sample.raw_input)
    return build_chat_prompt(sample.history, sample.item)


def load_llama_lora_test_samples(
    json_path: Path, limit: Optional[int] = None,
) -> List[Sample]:
    """Charge `llama_test.json` (format Alpaca `{instruction, input, output}`).

    Utilise POUR ECR[Llama 2-Chat] LoRA uniquement. Le format est different
    du dataset zero-shot (qui passe par `load_test_samples` sur
    `test_data_dbpedia_emo.jsonl`). Le `input` contient :
        Movie Name: <name>
        Related Entities: <ents>
        Related Knowledge from KG: <kg>
        First Sentence: <first sentence>
    Le modele LoRA est entraine a CONTINUER la First Sentence (pas a
    repondre "from scratch"). On extrait le nom du film via regex pour
    populer `Sample.item`, et on stocke le `input` complet dans
    `Sample.raw_input` pour reconstruire le prompt via `build_lora_prompt`
    (meme template que training -> zero mismatch).
    """
    import re as _re
    path = Path(json_path)
    with path.open() as f:
        raw = json.load(f)

    samples: List[Sample] = []
    for idx, row in enumerate(raw):
        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        gt = row.get("output") or None
        m = _re.search(r"Movie Name:\s*(.+)", input_text)
        item = m.group(1).strip() if m else ""
        samples.append(Sample(
            dialogue_id=str(row.get("id", idx)),
            history=input_text,  # passe au scorer comme contexte lisible
            item=item,
            ground_truth=(gt.strip() if isinstance(gt, str) and gt.strip() else None),
            instruction=instruction,
            raw_input=input_text,
        ))
        if limit and len(samples) >= limit:
            break
    return samples


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


def _dtype_kwarg_name():
    """Retourne le nom du kwarg dtype supporte par la version de transformers.

    `torch_dtype` etait le nom historique ; transformers 4.48+ a introduit le
    nom canonique `dtype` (avec `torch_dtype` deprecie, emission d'un
    `DeprecationWarning`). Comme les deux sont passes via `**kwargs` (pas de
    signature explicite), on inspecte le code source de
    `PreTrainedModel.from_pretrained` pour detecter la presence du pop de
    `dtype`. Cache le resultat au module-level (une seule fois par process).
    """
    global _DTYPE_KWARG_CACHE
    try:
        return _DTYPE_KWARG_CACHE  # type: ignore[name-defined]
    except NameError:
        pass
    import inspect
    try:
        from transformers import PreTrainedModel
        src = inspect.getsource(PreTrainedModel.from_pretrained)
        # La canonicalisation "dtype" a ete introduite en ~4.48. Si le pop
        # explicite est present, on est sur une version qui deprecate
        # `torch_dtype` -- on utilise directement `dtype`.
        if 'kwargs.pop("dtype"' in src or "kwargs.pop('dtype'" in src:
            _DTYPE_KWARG_CACHE = "dtype"
        else:
            _DTYPE_KWARG_CACHE = "torch_dtype"
    except (ImportError, OSError, TypeError):
        _DTYPE_KWARG_CACHE = "torch_dtype"
    return _DTYPE_KWARG_CACHE


def _load_model_with_fa2(model_dir, torch_dtype, device_map, token=None,
                          quantization_config=None, use_flash_attn_2=True):
    """Charge un modele HF en essayant Flash-Attention 2 d'abord, avec fallback sdpa.

    Flash-Attention 2 apporte 2-3x sur l'attention Llama 2 (sans impact qualite).
    Si le package `flash-attn` n'est pas installe ou pas compatible sm_120,
    on retombe sur `sdpa` (PyTorch 2.x Scaled Dot-Product Attention), qui est
    deja 1.3-1.5x plus rapide que l'impl eager.
    """
    from transformers import AutoModelForCausalLM
    dtype_key = _dtype_kwarg_name()
    kwargs = {dtype_key: torch_dtype, "device_map": device_map}
    if token is not None:
        kwargs["token"] = token
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs.pop(dtype_key, None)
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
    """Inference via transformers (bf16). Support LoRA via peft + Flash-Attn 2.

    IMPORTANT : cette fonction charge Llama-2-7B (12.5 GB bf16) dans le process
    Python appelant. On fait un `del model; del tokenizer; gc.collect();
    torch.cuda.empty_cache()` EXPLICITE avant le return pour minimiser les refs
    residuelles. NOTE : le CUDA context Python (~400-500 MB) persiste jusqu'a
    la fin du process, empty_cache ne peut pas le recuperer. Si la VRAM doit
    etre 100% rendue au driver (ex: avant de lancer Qwen 32B), il faut soit
    redemarrer le kernel Python, soit encapsuler cet appel dans un subprocess
    dedie (voir runbook Niveau 2).
    """
    import gc
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

    try:
        outputs = []
        for s in samples:
            # Dispatch selon type de sample : LoRA (instruction/raw_input) ou
            # zero-shot (history/item). Garantit zero mismatch train/infer
            # pour le LoRA, qui produisait des placeholders non resolus quand
            # on le drivait avec build_chat_prompt (bug initial).
            prompt = build_prompt_for_sample(s)
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
    finally:
        # Nettoyage explicite : del avant gc pour que les tensors referencent
        # plus de poids -> caching allocator libere au maximum.
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def generate_vllm(
    samples: List[Sample],
    base_url: str = "http://localhost:8001/v1",
    model_name: str = "NousResearch/Llama-2-7b-chat-hf",
    api_key: str = "EMPTY",
    max_new_tokens: int = 150,
    temperature: float = 0.7,
) -> List[Dict]:
    """Inference via vLLM (API OpenAI-compatible).

    On utilise `/v1/completions` (prompt brut, pas de templating cote serveur)
    plutot que `/v1/chat/completions` (messages -> apply_chat_template).

    Raison : le tokenizer de Llama-2-chat (officiel Meta ET miroir
    NousResearch) ne definit PAS de `chat_template` dans tokenizer_config.json.
    Le fallback transformers a ete retire en 4.44, donc vLLM renvoie un
    HTTP 400 "default chat template is no longer allowed" sur tout appel
    chat. En pre-formatant le prompt cote client via `build_chat_prompt`
    (template officiel Meta `<s>[INST]<<SYS>>...[/INST]`), on contourne ce
    check ET on garantit la coherence bit-a-bit avec le path HF
    (`generate_hf` qui utilise deja la meme fonction).

    `stop=["</s>", "[INST]"]` coupe la generation si Llama tente d'ouvrir un
    nouveau tour (eviter le sur-completion typique des LLM chat).
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    outputs = []
    for s in samples:
        # Meme dispatcher que generate_hf : LoRA ou zero-shot automatiquement.
        prompt = build_prompt_for_sample(s)
        resp = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["</s>", "[INST]"],
        )
        outputs.append({
            "dialogue_id": s.dialogue_id,
            "item": s.item,
            "response": resp.choices[0].text.strip(),
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
