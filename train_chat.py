#!/usr/bin/env python3
"""
train_chat.py — Chatbot / RP LoRA fine-tune on extracted MC dialogue data.

This is a second-pass fine-tune designed to teach an existing model (Sunfall,
Dolphin, or any Llama-3 base) to respond conversationally as the submissive/subject
character in SillyTavern — reacting to the player who takes the dominant role.

Differences from train.py (story fine-tune):
  - Shorter sequences (2048 tokens — conversation turns, not full stories)
  - Smaller dataset (10k–50k turns vs 17k full stories)
  - Can run on the RTX 3070 Ti locally — no A100 rental needed
  - More epochs (3–4) since dataset is smaller
  - Lower LoRA rank (r=8) — style adaptation, not full domain learning

Recommended base models (in order of preference):
  Vdr1/L3-8B-Sunfall-v0.5-Stheno-v3.2      ← best prose, what you're using
  cognitivecomputations/dolphin3.0-llama3.1-8b
  meta-llama/Meta-Llama-3.1-8B-Instruct

Usage:
  python extract_chatbot_data.py              # prepare data first
  python train_chat.py --train                # train on 3070 Ti (~2–4 hours)
  python train_chat.py --train --resume       # resume from checkpoint
  python train_chat.py --merge                # merge into full model
  python train_chat.py --test                 # quick generation test
  python train_chat.py --train --preset a100  # faster run on A100 rental
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

PRESETS = {
    "3070": {
        # RTX 3070 Ti — 8GB VRAM, ~2–4 hours
        "max_seq_length": 2048,
        "batch_size":     1,
        "grad_accum":     8,
        "lora_r":         8,
        "epochs":         3,
        "learning_rate":  1e-4,   # lower LR for style adaptation
    },
    "a100": {
        # A100 80GB — faster, higher quality (~45 min, ~$1–2)
        "max_seq_length": 4096,   # longer context = multi-turn conversations fit whole
        "batch_size":     4,
        "grad_accum":     4,
        "lora_r":         16,
        "epochs":         3,
        "learning_rate":  1e-4,
    },
    "mac": {
        # Mac M4 16GB via MLX — use mlx_lm.lora instead of this script
        # This preset is here for reference only; MLX has its own training CLI.
        "max_seq_length": 2048,
        "batch_size":     1,
        "grad_accum":     4,
        "lora_r":         8,
        "epochs":         3,
        "learning_rate":  1e-4,
    },
}

CONFIG = {
    # ── Model ──────────────────────────────────────────────────────────────────
    # Sunfall is the recommended base — it already has strong RP instincts.
    # The chat LoRA teaches it to stay in first-person conversational mode.
    "base_model":   "Vdr1/L3-8B-Sunfall-v0.5-Stheno-v3.2",

    # ── Paths ──────────────────────────────────────────────────────────────────
    "train_data":   "./output/chat_data.jsonl",
    "val_data":     "./output/chat_val.jsonl",     # held-out stories, from extract_chatbot_data.py
    "output_dir":   "./output/chat_lora",
    "merged_dir":   "./output/chat_merged",

    # ── Training ───────────────────────────────────────────────────────────────
    "max_seq_length": 2048,
    "batch_size":     1,
    "grad_accum":     8,
    "epochs":         3,
    "learning_rate":  1e-4,
    "warmup_steps":   30,
    "save_steps":     100,
    "logging_steps":  10,

    # ── LoRA — smaller rank than story fine-tune ────────────────────────────
    # r=8 is enough for style/behaviour adaptation; r=16 for more capacity.
    "lora_r":       8,
    "lora_alpha":   16,
    "lora_dropout": 0.05,
    "lora_target":  ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
}

SYSTEM_PROMPT = (
    "You are the subject in an erotic mind control roleplay. You respond as the person "
    "being hypnotized or controlled — vulnerable, reactive, and increasingly yielding. "
    "Speak in first person. Show internal conflict, physical sensation, and gradual "
    "submission. React directly to what the other person says or does. Your responses "
    "are 2–5 sentences. Never narrate in third person, never break character."
)


# ── Training ──────────────────────────────────────────────────────────────────

def _bfloat16_supported() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except Exception:
        return False


def train(resume: bool = False):
    """Train the chat LoRA on chat_data.jsonl with Unsloth (4-bit QLoRA), evaluating on
    chat_val.jsonl when present. Checkpoints go to CONFIG['output_dir']; --resume
    continues from the last one; final weights land in output_dir/final.
    """
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments, TrainerCallback
        from datasets import load_dataset
    except ImportError as e:
        print(f"\nERROR: Missing dependency: {e}")
        print("pip install unsloth trl datasets transformers accelerate bitsandbytes")
        sys.exit(1)

    data_path = Path(CONFIG["train_data"])
    if not data_path.exists():
        print(f"No training data at {data_path}")
        print("Run: python extract_chatbot_data.py")
        sys.exit(1)

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n_lines = sum(1 for _ in open(data_path))
    size_mb = data_path.stat().st_size / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"  MC Chatbot LoRA Fine-Tune")
    print(f"  Model:  {CONFIG['base_model']}")
    print(f"  Data:   {data_path}  ({n_lines:,} conversations, {size_mb:.0f} MB)")
    print(f"  Output: {out_dir}")
    print(f"  LoRA:   r={CONFIG['lora_r']}  lr={CONFIG['learning_rate']}")
    print(f"{'='*60}\n")

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = CONFIG["base_model"],
        max_seq_length = CONFIG["max_seq_length"],
        dtype          = None,
        load_in_4bit   = True,
    )

    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r              = CONFIG["lora_r"],
        target_modules = CONFIG["lora_target"],
        lora_alpha     = CONFIG["lora_alpha"],
        lora_dropout   = CONFIG["lora_dropout"],
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({100*trainable/total:.1f}%)\n")

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    print(f"Dataset: {len(dataset):,} examples")

    def format_example(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_example, batched=False)

    # Held-out stories (written by extract_chatbot_data.py). Validation loss shows
    # whether the model generalises to unseen stories; training loss can't.
    val_path = Path(CONFIG["val_data"])
    eval_dataset, eval_kwargs = None, {}
    if val_path.exists() and val_path.stat().st_size > 0:
        eval_dataset = load_dataset("json", data_files=str(val_path), split="train")
        eval_dataset = eval_dataset.map(format_example, batched=False)
        print(f"Validation: {len(eval_dataset):,} examples (held-out stories)")
        import inspect
        # transformers renamed evaluation_strategy → eval_strategy (4.41+)
        strategy_arg = ("eval_strategy" if "eval_strategy" in
                        inspect.signature(TrainingArguments.__init__).parameters
                        else "evaluation_strategy")
        eval_kwargs = {strategy_arg: "steps", "eval_steps": CONFIG["save_steps"],
                       "per_device_eval_batch_size": CONFIG["batch_size"]}
    else:
        print("No validation data — re-run extract_chatbot_data.py to create chat_val.jsonl")

    steps_per_epoch = max(1, len(dataset) // (CONFIG["batch_size"] * CONFIG["grad_accum"]))
    total_steps     = steps_per_epoch * CONFIG["epochs"]

    # Simple loss tracker
    losses = []
    best   = float("inf")
    t0     = time.time()

    class LossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            nonlocal best
            if logs and "eval_loss" in logs:
                print(f"  Step {state.global_step:>5}  validation loss: {logs['eval_loss']:.4f}")
            if not logs or "loss" not in logs:
                return
            loss = logs["loss"]
            losses.append(loss)
            best = min(best, loss)
            elapsed = time.time() - t0
            rate    = max(state.global_step, 1) / elapsed
            eta_s   = (total_steps - state.global_step) / rate
            eta_m   = int(eta_s // 60)
            spark   = ""
            if len(losses) > 1:
                lo, hi = min(losses[-30:]), max(losses[-30:])
                chars  = " ▁▂▃▄▅▆▇█"
                spark  = "".join(chars[min(8, int(8*(v-lo)/max(hi-lo, 0.001)))]
                                 for v in losses[-30:])
            print(f"  Step {state.global_step:>5}/{total_steps}  "
                  f"Loss: {loss:.4f}  Best: {best:.4f}  "
                  f"ETA: {eta_m//60}h{eta_m%60:02d}m  |{spark}|")

    args = TrainingArguments(
        output_dir                  = str(out_dir),
        num_train_epochs            = CONFIG["epochs"],
        per_device_train_batch_size = CONFIG["batch_size"],
        gradient_accumulation_steps = CONFIG["grad_accum"],
        warmup_steps                = CONFIG["warmup_steps"],
        learning_rate               = CONFIG["learning_rate"],
        fp16                        = not _bfloat16_supported(),
        bf16                        = _bfloat16_supported(),
        logging_steps               = CONFIG["logging_steps"],
        save_steps                  = CONFIG["save_steps"],
        save_total_limit            = 3,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = 42,
        report_to                   = "none",
        resume_from_checkpoint      = resume,
        **eval_kwargs,
    )

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = dataset,
        eval_dataset       = eval_dataset,
        dataset_text_field = "text",
        max_seq_length     = CONFIG["max_seq_length"],
        args               = args,
        callbacks          = [LossCallback()],
    )

    print(f"\nTraining: {CONFIG['epochs']} epochs × {steps_per_epoch} steps = {total_steps} total")
    print(f"Checkpoints → {out_dir}  (Ctrl+C safe, resume with --resume)\n")

    trainer.train(resume_from_checkpoint=resume)

    final = out_dir / "final"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))

    elapsed = int(time.time() - t0)
    print(f"\nDone in {elapsed//3600}h {(elapsed%3600)//60:02d}m")
    print(f"Best loss: {best:.4f}   Final loss: {losses[-1]:.4f}")
    print(f"LoRA weights → {final}")
    print(f"Next: python train_chat.py --merge")


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge():
    """Merge the trained LoRA (output_dir/final) into the base model and save the full
    model to CONFIG['merged_dir'].
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: pip install unsloth"); sys.exit(1)

    lora_path   = Path(CONFIG["output_dir"]) / "final"
    merged_path = Path(CONFIG["merged_dir"])

    if not lora_path.exists():
        print(f"No LoRA weights at {lora_path}. Run training first.")
        sys.exit(1)

    print(f"Merging LoRA into base model...")
    print(f"  Source: {lora_path}")
    print(f"  Output: {merged_path}  (~15GB, takes ~5 min)\n")

    merged_path.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(lora_path),
        max_seq_length = CONFIG["max_seq_length"],
        dtype          = None,
        load_in_4bit   = True,
    )
    model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")

    print(f"\nMerged → {merged_path}")
    print(f"\nConvert to GGUF (on Windows):")
    print(f"  git clone https://github.com/ggerganov/llama.cpp")
    print(f"  cd llama.cpp && pip install -r requirements.txt")
    print(f"  python convert_hf_to_gguf.py {merged_path} --outtype q4_k_m --outfile chat_model_Q4_K_M.gguf")
    print(f"\nLoad in LM Studio → set Prompt Format: Llama 3")
    print(f"Connect SillyTavern → your character card overrides the system prompt.")


# ── Test ───────────────────────────────────────────────────────────────────────

def test():
    """Load the trained LoRA and print its replies to a few fixed RP prompts, for a quick
    manual check.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: pip install unsloth"); sys.exit(1)

    lora_path = Path(CONFIG["output_dir"]) / "final"
    if not lora_path.exists():
        print(f"No model at {lora_path}"); sys.exit(1)

    print(f"Loading {lora_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(lora_path),
        max_seq_length = CONFIG["max_seq_length"],
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)

    test_conversations = [
        # Test 1: Subject pushed to relax — should show resistance fading
        [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Look into my eyes. Let your thoughts slow down. You're feeling very relaxed now."},
        ],
        # Test 2: Mid-trance command — should show compliance with internal conflict
        [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "That's right. Just follow my voice. You don't need to think anymore."},
            {"role": "assistant", "content": "I... I tried to hold onto a thought, but it slipped away before I could grasp it. Your voice was the only thing that felt solid."},
            {"role": "user",      "content": "Good. Now tell me what you're feeling."},
        ],
        # Test 3: Subject pushing back — should show reluctant surrender
        [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Stop fighting it. You know you want to let go."},
        ],
    ]

    import torch
    print(f"\n{'='*60}\nGENERATION TEST\n{'='*60}")

    for i, msgs in enumerate(test_conversations, 1):
        print(f"\n--- Prompt {i} ---")
        print(msgs[-1]["content"])
        print(f"\n--- Response {i} ---")

        text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens     = 300,
                temperature        = 0.9,
                top_p              = 1.0,
                min_p              = 0.05,
                repetition_penalty = 1.1,
                do_sample          = True,
                pad_token_id       = tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(response[:800])
        if len(response) > 800:
            print(f"[...{len(response)} chars total]")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    """Command-line entry point: --train [--resume], --merge, --test, --preset."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train",  action="store_true", help="Run training")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    ap.add_argument("--merge",  action="store_true", help="Merge LoRA into full model")
    ap.add_argument("--test",   action="store_true", help="Test generation")
    ap.add_argument("--preset", choices=list(PRESETS), default=None,
                    help="Hardware preset (3070 / a100 / mac)")
    ap.add_argument("--model",  default=None,
                    help="Override base model (HuggingFace repo ID)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    args = ap.parse_args()

    if args.preset:
        CONFIG.update(PRESETS[args.preset])
        print(f"Using preset: {args.preset}")

    if args.model:
        CONFIG["base_model"] = args.model
    if args.epochs:
        CONFIG["epochs"] = args.epochs
    if args.lora_r:
        CONFIG["lora_r"] = args.lora_r
    if args.learning_rate:
        CONFIG["learning_rate"] = args.learning_rate

    if not any([args.train, args.merge, args.test]):
        ap.print_help()
        sys.exit(0)

    if args.train:
        train(resume=args.resume)
    if args.merge:
        merge()
    if args.test:
        test()


if __name__ == "__main__":
    main()
