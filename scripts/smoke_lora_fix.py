"""Smoke test post-fix: valide template LoRA + dispatcher + shims."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/workspace")


def main():
    # --- 1. llama_lora template + build_lora_prompt ---
    from src.ecr.llama_lora import PROMPT_TEMPLATE, build_lora_prompt, build_training_examples
    print("=== llama_lora ===")
    print("PROMPT_TEMPLATE:", repr(PROMPT_TEMPLATE[:80]))
    p = build_lora_prompt(
        "You are a recommender...",
        "Movie Name: Inception\nFirst Sentence: Great movie",
    )
    print("build_lora_prompt sample (100 chars):", repr(p[:100]))

    # --- 2. build_training_examples sur le vrai llama_train.json ---
    for path in ("/workspace/data/emo_data/llama_train.json",
                 "/workspace/ECR/src_emo/data/redial_gen/llama_train.json"):
        if Path(path).exists():
            examples = build_training_examples(Path(path))
            print(f"  {path}: {len(examples)} valid examples")
            if examples:
                sp = examples[0]["prompt"]
                st = examples[0]["target"]
                print(f"  sample[0] prompt[:100]: {sp[:100]!r}")
                print(f"  sample[0] target[:80]:  {st[:80]!r}")
            break

    # --- 3. Sample dataclass with new fields ---
    from src.ecr.llama_runner import Sample, build_prompt_for_sample
    print()
    print("=== Sample dispatch ===")
    s_zeroshot = Sample(dialogue_id="1", history="hi", item="Inception")
    s_lora = Sample(dialogue_id="2", history="ctx", item="Titanic",
                    instruction="You are a recommender...",
                    raw_input="Movie Name: Titanic\n...")
    p_zs = build_prompt_for_sample(s_zeroshot)
    p_lo = build_prompt_for_sample(s_lora)
    assert "Recommended movie: Inception" in p_zs, "zero-shot dispatch broke"
    assert "Movie Name: Titanic" in p_lo, "lora dispatch broke"
    assert "Recommended movie:" not in p_lo, "lora should NOT use zero-shot template"
    print("  zero-shot dispatch OK")
    print("  lora dispatch OK")

    # --- 4. load_llama_lora_test_samples ---
    from src.ecr.llama_runner import load_llama_lora_test_samples
    samples = load_llama_lora_test_samples(
        Path("/workspace/ECR/src_emo/data/redial_gen/llama_test.json"), limit=3,
    )
    print()
    print("=== load_llama_lora_test_samples ===")
    for s in samples[:2]:
        print(f"  dialogue_id={s.dialogue_id}")
        print(f"  item={s.item!r}")
        print(f"  instruction[:60]={(s.instruction or '')[:60]!r}")
        print(f"  raw_input[:80]={(s.raw_input or '')[:80]!r}")

    # --- 5. Verify shim + dtype detection ---
    from src.ecr.llama_lora import _ensure_accelerator_unwrap_compat
    from src.ecr.llama_runner import _dtype_kwarg_name
    _ensure_accelerator_unwrap_compat()
    print()
    print("=== runtime shims ===")
    print("dtype_kwarg:", _dtype_kwarg_name())

    # --- 6. scorer kill_vllm_proc_tree imports ---
    from src.ecr.scorer import kill_vllm_proc_tree, launch_vllm_server, wait_vllm_ready
    print("scorer helpers importable: OK")

    print()
    print("=== ALL CHECKS PASS ===")


if __name__ == "__main__":
    main()
