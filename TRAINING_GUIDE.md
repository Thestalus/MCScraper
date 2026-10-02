# Complete Training Guide
## Fine-tuning for Story Generation AND Chatbot / SillyTavern RP

> **Document status:** Working personal reference
> **Version:** 2.0
> **Last updated:** 2026-07-13
> **Primary hardware:** RTX 3070 Ti 8 GB (Windows), Mac M4 16 GB
> **Current recommended target:** 12B fine-tuned (Magnum or Dolphin Nemo), Q4_K_S on 3070 Ti
> **Validated environment:** NOT YET VALIDATED end-to-end on 12B — see Known Facts section

---

### Evidence labels

Used throughout this document to distinguish confidence levels:

- **[Measured]** Personally observed in a completed run.
- **[Calculated]** Derived from measured throughput, token counts, or VRAM figures.
- **[Estimated]** Plausible planning figure not yet validated end to end.
- **[Community]** Based on community reports rather than personal testing.
- **[Subjective]** Personal quality judgement.
- **[Volatile]** Likely to change with software updates, pricing, or newer models.

Labels appear inline as **[Measured]**, **[Estimated]**, etc. wherever the guide states performance, cost, quality, timing, coherence, or hardware-fit claims.

---

## Overview

There are two separate fine-tune pipelines. You can run either or both:

### Pipeline A — Story Generation (original)
Trains the model to write long-form MC erotica on demand. Full workflow:

1. **Windows** — validate the pipeline works locally on a tiny dataset (free, ~30 min)
2. **Mac** — prepare and filter your training data
3. **A100 rental** — run the real training (2–5 hours, ~$2–4)
4. **Mac M4** — download the model, convert to GGUF, run in LM Studio

### Pipeline B — Chatbot / SillyTavern RP (new)
Trains the model to respond conversationally as a dominant character — first person,
reactive, correct length. 12B chat training on your RTX 3070 Ti is tight but feasible
(~8–10 hours); using a Vast.ai A100 is faster and safer (~1.5 hours, ~$1–2).

1. **Mac** — extract dialogue data from stories (`extract_chatbot_data.py`)
2. **Windows** — train chatbot LoRA (`train_chat.py --train`, 2–4 hours)
3. **Windows** — merge and convert to GGUF
4. **Mac** — load in LM Studio, connect SillyTavern

**Recommended base model:** `anthracite-org/magnum-v4-12b` or `cognitivecomputations/dolphin3.0-mistral-nemo-12b` — 12B is the minimum viable size for RP with progressive mechanics. The chatbot LoRA teaches conversational behaviour on top. If your 3070 Ti consistently OOMs during 12B training, see the 8B fallback note in the 12B Model Option section.

---

### Why two pipelines?

Story generation and chatbot RP are different skills:

| | Story fine-tune | Chatbot fine-tune |
|---|---|---|
| Training data | Full stories (2k–40k words) | Dialogue turns (50–300 words) |
| What it learns | Genre vocabulary, prose style, arc structure | First-person response, reactive dialogue, scene continuation |
| Sequence length | 16,384 tokens | 2,048 tokens |
| Hardware needed | A100 (too slow on 3070 Ti) | RTX 3070 Ti (fits comfortably) |
| Training time | 2–14 hours on A100 | 2–4 hours on 3070 Ti |
| Cost | ~$2–6 (A100 rental at current prices) | Free (your own GPU) |
| Output | Model that writes stories | Model that RPs conversations |

The chatbot fine-tune can be applied on top of the story fine-tune (stack them), or
applied directly to a base model like Sunfall. Either approach works.

---

### Choosing a base model — why it matters

LoRA fine-tuning adds adapter layers on top of an existing model's **frozen** weights.
Your training data teaches only the genre-specific delta — everything else stays as it
was in the base. This means the quality of your starting point matters a lot.

A raw base model (e.g. Llama 3.1 8B before any instruction tuning) has no instruction
following, no RP behaviour, and no content permissions. Your fine-tune would waste most
of its budget re-teaching things a pre-tuned model already knows. Starting from a model
that has already been fine-tuned for RP or fiction means your LoRA only needs to add
the genre-specific layer on top of working RP fundamentals.

**The stacking order:**

```
Raw base model
    → Instruction tuning + RLHF       (e.g. Llama 3.1 Instruct, Qwen2.5 Instruct)
        → Content removal              (e.g. Dolphin — uncensored instruct)
            → RP / prose quality       (e.g. Sunfall, Magnum — character voice, prose variety)
                → Your genre LoRA      ← this guide
```

This guide uses Dolphin as the Pipeline A base (uncensored, strong instruction following —
layer 3) and Sunfall as the Pipeline B base (RP character voice already established —
layer 4). For 32B, Magnum by Anthracite is purpose-built for RP and creative writing,
designed to be fine-tuned further, and consistently outperforms bare Qwen-32B-Instruct
as a starting point because prose quality and character voice are already in its frozen weights.

**Recommended bases by size:**

| Size | Pipeline A (story fine-tune) | Pipeline B (chatbot LoRA) |
|---|---|---|
| 8B | `cognitivecomputations/dolphin3.0-llama3.1-8b` | `Vdr1/L3-8B-Sunfall-v0.5-Stheno-v3.2` |
| 12B | `cognitivecomputations/dolphin3.0-mistral-nemo-12b` | Same, or `anthracite-org/magnum-v4-12b` |
| 32B | `anthracite-org/magnum-v4-32b` | `anthracite-org/magnum-v4-32b` |

Check r/SillyTavernAI and r/LocalLLaMA before running — the best available RP base
changes every few months, and there may be a newer Magnum version or community favourite
worth using instead.

---

## Authoritative Model Matrix

Single reference for all model choices. Every later section refers back here rather than independently naming a preferred model. Check r/LocalLLaMA and r/SillyTavernAI before running — the best RP bases change every few months.

| Use case | Preferred model | Fallback | Reason | Prompt template | Status |
|---|---|---|---|---|---|
| Pipeline validation | `cognitivecomputations/dolphin3.0-llama3.1-8b` | Any uncensored 8B instruct | Free local run, 3070 Ti fits easily | ChatML | [Measured] — validation confirmed |
| Story LoRA 12B | `anthracite-org/magnum-v4-12b` | `cognitivecomputations/dolphin3.0-mistral-nemo-12b` | Magnum has strongest prose base before LoRA | Mistral Nemo | [Estimated] — not yet trained |
| Chat LoRA 12B | `anthracite-org/magnum-v4-12b` | `cognitivecomputations/dolphin3.0-mistral-nemo-12b` | Same base for both adapters — simplifies stacking | Mistral Nemo | [Estimated] — not yet trained |
| Arc classification | `cognitivecomputations/dolphin3.0-llama3.1-8b` fine-tuned | Base 8B instruct | Needs 32K ctx; 12B won't fit in 8GB VRAM at that ctx | ChatML | [Estimated] |
| 32B quality path | `anthracite-org/magnum-v4-32b` | `Qwen/Qwen2.5-32B-Instruct` | Strongest prose/voice base; designed for further fine-tuning | Magnum/Qwen | [Estimated] — separate project |

**Terminology used in this guide:**
- **Foundation model** — raw pretrained weights (e.g. Llama 3.1, Mistral Nemo base)
- **Instruction-tuned model** — foundation + supervised instruction following (e.g. Llama 3.1 Instruct)
- **RP/prose-tuned base** — instruction-tuned + RP/prose fine-tuning (e.g. Magnum, Sunfall, Dolphin)
- **Story LoRA** — your adapter trained on full stories; adds genre vocabulary and arc structure
- **Chat LoRA** — your adapter trained on extracted dialogue; adds first-person reactive behaviour
- **Merged model** — RP base with one or both LoRAs merged into full weights
- **Deployment model** — merged model quantized to GGUF for inference

> **Note on Sunfall:** Earlier versions of this guide used Sunfall 8B as the Pipeline B base. Sunfall is an 8B model. The current recommendation is Magnum 12B for both pipelines. Sunfall settings (Llama 3 template, Q5_K_M) remain in the 8B fallback sections.

---

You never need to train on the Mac. The Mac is for running the finished model and for
scraping/data prep. Heavy inference tasks (arc tagging 17k stories) run better on
Windows — see the Hardware Notes section below.

---

## Current Recommended Path

Answers the key decisions without reading the full guide. All detail is in later sections.

| Question | Answer |
|---|---|
| **Which pipeline first?** | Pipeline B (chat LoRA) — cheaper, faster feedback, works on 3070 Ti |
| **Preferred base model?** | See Authoritative Model Matrix below |
| **Minimum viable size?** | 12B — 8B does not meet the bar for progressive RP mechanics |
| **Training GPU?** | Vast.ai A100 80GB for 12B story LoRA; 3070 Ti for 12B chat LoRA (tight) |
| **Deployment quant?** | Q4_K_S on 3070 Ti; Q4_K_M on Mac only |
| **Context length?** | 6144 on 3070 Ti (12B); 8192 on Mac |
| **Where to run inference?** | 3070 Ti via LAN to Mac SillyTavern |
| **When to run arc tagging?** | After story LoRA is working; use 8B for classification at 32K ctx |
| **When to do second training run?** | After evaluating chat LoRA; only spend on story LoRA if chat quality justifies it |
| **8B fallback?** | If 12B QLoRA consistently OOMs on 3070 Ti — acceptable for story drafts only |

> **Personal conclusion:** Train 12B chat LoRA first (~$1–2 on A100). Evaluate before spending on story training. 32B is a separate hardware-upgrade project, not a natural next step.

---

## Known Facts and Open Questions

Tracks what has been validated versus what is estimated or untested. Prevents figures from hardening into facts across edits.

### Confirmed

- Pipeline A (story LoRA) validation run completes on RTX 3070 Ti with 8B Dolphin at short context. **[Measured]**
- 8B GGUF inference in LM Studio on Mac M4: ~17 tok/s generation, ~174 tok/s prompt eval. **[Measured]**
- 12B Q4_K_S weights: ~6.5 GB. 12B Q4_K_M weights: ~7.0 GB. **[Calculated]**
- Arc tagging at 32K context requires 8B — 12B does not fit in 8GB VRAM at that context. **[Calculated]**
- OurDream base tier includes auto-summarisation and optional pinned memories. **[Confirmed from samples]**
- 8B models produce binary resist/yield emotional gradation; do not maintain progressive state across sessions. **[Subjective, from OurDream sample comparison]**

### Not Yet Validated

- Full 12B chat QLoRA training stability on RTX 3070 Ti (7–7.5 GB estimate is tight).
- Actual A100 training duration for 12B at 16K context on full 17K-story corpus.
- Sequential adapter stacking (story LoRA merged → chat LoRA trained on merged model).
- Whether Magnum 12B or Dolphin Nemo 12B produces better RP output for this corpus.
- 12B Q4_K_S inference speed at 6144 context on 3070 Ti (estimated ~30–40 tok/s).
- Whether story LoRA degrades chat behaviour (overlong replies, third-person narration).
- Whether chat LoRA degrades story generation (short outputs, dialogue-heavy prose).
- Quality benefit of 32K over 16K training context for story LoRA.
- Arc-tag pseudo-label accuracy (not benchmarked against human labels).
- Whether arc-conditioned retraining improves story generation in practice.

---

## What you'll end up with

A fine-tuned 8B model that:
- Writes in the mc erotica genre's style natively
- Follows your tag prompts (mc_type, setting, themes, dominance, awareness, etc.)
- Has no content restrictions
- Runs locally in LM Studio on your M4 at ~50-60 tok/sec (interactive use)
- Arc tagging runs better on Windows RTX 3070 Ti (~25 sec/story vs ~60 sec on Mac)
- Costs nothing to run after the one-time training fee (~$2–4 at current A100 prices)

---

## Adapter Stacking — How the Two LoRAs Combine

The guide supports two pipelines that can be used independently or combined. The stacking method matters for merge order, template consistency, and output behaviour.

```text
Option A — Independent adapters (run only one)
RP/prose base (e.g. Magnum 12B)
├── Story LoRA only  →  best for long-form story generation
└── Chat LoRA only   →  best for SillyTavern RP, minimum viable setup

Option B — Sequential stacking (run both, recommended order)
RP/prose base (e.g. Magnum 12B)
→ Story LoRA trained and MERGED into full model weights
→ Chat LoRA trained on the merged story model
→ Final merge produces: prose base + story knowledge + chat behaviour
```

**Option A — Independent adapters:**
- Supported by current scripts with no changes.
- Story LoRA and Chat LoRA are trained separately from the same RP base.
- Either adapter is merged alone for deployment.
- Advantage: clean separation, easy to compare.
- Risk: Chat LoRA does not benefit from story-domain genre vocabulary.
- Recommended when: evaluating chat behaviour before committing to story training.

**Option B — Sequential stacking (story → chat):**
- Train story LoRA first, merge it into the RP base (`python train.py --merge`).
- Then train chat LoRA using the merged story model as the starting point (`python train_chat.py --train --model ./output/merged_model`).
- Final merge produces the combined model.
- Advantage: Chat LoRA builds on genre vocabulary from story training — richer responses.
- Risk: If story LoRA causes prose drift (third-person narration in chat, overlong replies), the chat LoRA inherits this. Evaluate story output before proceeding.
- Recommended when: story LoRA quality is confirmed and chat behaviour is the final target.

**What does NOT work:**
- Loading both adapters simultaneously at inference without merging — standard GGUF runtime does not support this.
- Training chat LoRA without merging story LoRA first — the story adapter would be ignored.

**Template consistency:**
Both LoRAs must use the same prompt template as the RP base they are built on. If Magnum 12B uses the Mistral Nemo template, all training and inference must use that template. Switching templates between training and inference causes role leakage or garbage output.

> **NOT YET VALIDATED:** Sequential stacking (Option B) is the intended approach but has not been run end-to-end. Validate Option A (chat LoRA only) first before committing to Option B.

---

## Prompt Template Reference

**Critical:** Training with one template and inferring with another produces role leakage, repeated headers, or garbage output. Verify this table against the model card before training.

| Model | Template name | Training format | SillyTavern preset | LM Studio setting | BOS/EOS | Notes |
|---|---|---|---|---|---|---|
| Magnum 12B / 32B | Mistral Nemo | Mistral Nemo chat template | Mistral | Mistral Nemo | `<s>` / `</s>` | [Community] — verify against current model card |
| Dolphin Nemo 12B | ChatML | `<\|im_start\|>` / `<\|im_end\|>` | ChatML | ChatML | `<s>` / `</s>` | [Community] — Dolphin overrides Nemo template with ChatML |
| Dolphin 8B (Llama 3.1) | ChatML | `<\|im_start\|>` / `<\|im_end\|>` | ChatML | Llama 3 (ChatML mode) | `<\|begin_of_text\|>` | [Measured] — used in validation |
| Sunfall 8B | Llama 3 | Llama 3 instruct | Llama 3 | Llama 3 | `<\|begin_of_text\|>` | [Community] |
| Magnum 32B | Mistral Nemo | Mistral Nemo chat template | Mistral | Mistral Nemo | `<s>` / `</s>` | [Community] |

**Verification command** (run before training):
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("model-name")
print(tok.chat_template)
```

**Symptoms of template mismatch:**
- Output contains `<|im_start|>`, `[INST]`, or `### Human:` text in generated prose
- Model repeats the user's message back
- Generation never stops (missing EOS)
- Every response begins with "As an AI..."
- Output is coherent but clearly not following instructions

> **VERIFY IN SCRIPT:** Confirm which template `train.py` and `train_chat.py` apply to each preset. If Magnum 12B uses a non-ChatML template, the `a100_12b` preset must serialize accordingly.

---

## Canonical Quantization and Context Reference

Single source of truth for quantization and context settings. Resolves conflicting recommendations elsewhere in this guide. VRAM figures are **[Estimated]** from architecture specs and community reports — measure actual peak VRAM in the runtime before relying on tight fits.

| Model | Runtime | Hardware | Quant | Weights | Context | KV cache (est.) | Total est. VRAM | Status |
|---|---|---|---|---|---|---|---|---|
| 12B (Nemo/Magnum) | LM Studio / KoboldCPP | RTX 3070 Ti (8GB) | Q4_K_S | ~6.5 GB | 6144 | ~1.0 GB | ~7.5 GB | **Recommended** [Estimated] |
| 12B (Nemo/Magnum) | LM Studio | Mac M4 16GB | Q4_K_M | ~7.0 GB | 6144 | ~1.0 GB | ~8.0 GB + OS | Travel fallback [Estimated] |
| 8B (Llama 3.1) | LM Studio / KoboldCPP | RTX 3070 Ti (8GB) | Q5_K_M | ~5.7 GB | 8192 | ~1.3 GB | ~7.0 GB | 8B fallback [Measured approx] |
| 8B (Llama 3.1) | LM Studio | Mac M4 16GB | Q4_K_M | ~4.5 GB | 8192 | ~1.3 GB | ~5.8 GB + OS | Mac fallback [Measured approx] |
| 8B (Llama 3.1) | LM Studio | RTX 3070 Ti (8GB) | Q4_K_S | ~4.7 GB | 32768 | ~1.0 GB | ~5.7 GB | Arc tagging only [Estimated] |
| 32B (Magnum) | KoboldCPP | Vast.ai 3090 (24GB) | Q4_K_M | ~18 GB | 8192 | ~2.5 GB | ~20.5 GB | 32B path [Estimated] |

**Notes:**
- Q4_K_M must NOT be used for 12B on the 3070 Ti: weights (~7.0GB) + KV cache overflows 8GB VRAM.
- Q5_K_M is preferred over Q4_K_M for 8B on the 3070 Ti — better quality, still fits.
- Mac context should be 6144 for 12B (not 8192) — 16GB unified memory is shared with OS and other apps.
- SillyTavern Max Context should be set 500–700 tokens below the server context to leave generation headroom.
- These VRAM figures do not include CUDA context, compute buffers, or display GPU usage. Measure actual peak in `nvidia-smi` during a real session before calling a configuration stable.

> **Unresolved:** Q4_K_S vs Q4_K_M quality difference for 12B has not been personally benchmarked. Use Q4_K_S on 3070 Ti as the safe choice regardless.

---

## Phase 1 — Windows validation (do this first, it's free)

The goal here is NOT good output — it's proving the pipeline works end to end
before spending money on the A100. You're checking CUDA works, Unsloth installs,
data format is correct, checkpoints save, merge works.

### 1.1 — Copy files to Windows

Copy these from your Mac to a folder on Windows (e.g. C:\MCTraining\):
- train.py
- tagger.py
- output/stories_index.json  (~500MB — use USB drive)
- output/preferences.json    (optional — your likes)

### 1.2 — Install CUDA on Windows

Check if already installed:
```cmd
nvcc --version
```
If not found, download CUDA Toolkit 12.1:
https://developer.nvidia.com/cuda-12-1-0-download-archive

Reboot after installing. Check again:
```cmd
nvcc --version
```

### 1.3 — Create Python environment

Open Command Prompt (not PowerShell):
```cmd
cd C:\MCTraining
python -m venv venv
venv\Scripts\activate
```
Prompt should now start with (venv).

### 1.4 — Install PyTorch with CUDA

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Downloads ~2GB. Verify GPU detected:
```cmd
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Should print:
```
True
NVIDIA GeForce RTX 3070 Ti
```
If False: check Device Manager for driver issues, try rebooting.

### 1.5 — Install training dependencies

```cmd
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install trl datasets transformers accelerate peft bitsandbytes sentencepiece protobuf
```
If the unsloth line fails, try: pip install unsloth

### 1.6 — Run validation training

Trains on 100 stories, 1 epoch, tiny context. Takes 20-40 minutes:
```cmd
python train.py --prepare --max-stories 100 --max-words 3000
python train.py --train --preset 3070 --epochs 1 --seq-len 512
```

Watch for these in the output:
- "Loading model" — downloads dolphin (~5GB, one time only)
- "Trainable params: X.XM / 8000M" — LoRA applied
- "Step X/X, Loss: X.XX" — training running
- Checkpoints saving

If OOM error: try --seq-len 256

### 1.7 — Test and validate merge

```cmd
python train.py --test
python train.py --merge
```
Output will be low quality (100 stories) but should generate text without crashing.
The merge creates output/merged_model/ (~15GB) — delete it after, it was just a test.

Pipeline is validated. You're ready for the real run.

---

## Phase 2 — Prepare real training data (on Mac)

### 2.1 — Make sure stories are tagged and index is clean

```bash
# Fix run-on word artifacts from HTML-stripping bug (e.g. "hishuge" → "his huge")
# Uses wordninja with guards for proper names, genre terms, and compound words.
# Also rebuilds the SQLite text cache.
make fix-text
# or: .venv/bin/python3 fix_text.py --dry-run   (preview first)

# Re-tag all stories (regex + semantic)
make tag
make tag-semantic
# Or equivalently in one command:
make post-update
```

`post-update` also rebuilds the JSONL and recs cache — you can skip it here
and just run the two `make tag` commands if you prefer.

### 2.2 — Like stories in the browser

```bash
make browse
```
Like stories and follow authors you enjoy. Even 50-100 likes helps.
Liked stories appear 3x in training, followed author stories 2x.
Disliked stories are excluded entirely.

You can like stories from the card ♡ button directly in search results and
author pages — no need to open each story individually.

### 2.3 — Prepare the data

```bash
# Standard A100 run — 16k context, ~15k stories, long ones chunked (~2-4 hours)
python train.py --preset a100 --chunk --prepare

# Maximum quality — 32k context, nearly all stories seen whole (~4-6 hours)
python train.py --preset a100_max --chunk --prepare

# Lower minimum word count if desired (default: 1500 words)
python train.py --preset a100 --chunk --min-words 1000 --prepare
```

The `a100` preset uses 16,384 token context (~12,000 words). Stories that fit
are included whole. Stories over ~10,650 words are split into overlapping chunks
so their full text is still seen during training. Previously, long stories were
excluded instead of chunked — this is now fixed. The preset covers ~15,000–17,000
stories (up from ~5,000 at the old settings).

Use `a100_max` for 32k context — almost no stories need chunking, and the model
sees nearly every story complete. Takes longer but produces the best arc-level
understanding.

> **Note:** Longer training contexts expose the model to more complete narrative trajectories, which may improve arc imitation — but this is a hypothesis, not a demonstrated improvement for this corpus. The practical preference for longer context is kept because the cost difference is modest; evaluate empirically on the first real run.

**Time and cost estimates (Vast.ai A100 40 GB ~$0.29–0.45/hr; A100 80 GB ~$0.45–0.70/hr):**

| Command | Stories | Records (weighted) | Est. time | Est. cost |
|---------|---------|-------------------|-----------|-----------|
| `--preset a100 --chunk` | ~15,000 | ~25,000–35,000 | 2–4 hours | $0.90–2.80 |
| `--preset a100_max --chunk` | ~17,000 | ~20,000–28,000 | 4–6 hours | $1.80–4.20 |
| `--preset a100` (no chunk) | ~14,000 | ~18,000–24,000 | 1.5–3 hours | $0.70–2.10 |

Example output after running:
```
Kept:          15,846
  too_short        176
  fetch_error      312
  no_themes      1,156

Weighting:
  Svengali-tagged stories: 4,821  (×2)
  Liked stories:              87  (×3)
  Readers' picks:            142  (×2)
  Followed author stories:   234  (×2)

Total records:  28,430  (after weighting + chunking)
File size:      3.8 GB
Est. tokens:    145.2M
Est. train time: 3.1 hours (A100 80GB)
```

This creates output/train_data.jsonl — this is what you upload to the A100.

---

## Phase 3 — A100 rental on Vast.ai

### 3.1 — Create account

Go to https://vast.ai, create account, add credit card.
Total cost will be ~$4-8.

### 3.2 — Find an A100 instance

1. Search tab
2. Filter: GPU Type = A100, VRAM >= 40GB (80GB preferred)
3. Sort by price — A100 40 GB should be $0.29–0.45/hr, A100 80 GB $0.45–0.70/hr
4. Look for: CUDA 12.1+, Disk >= 50GB, Verified badge
5. Click Rent

### 3.3 — Connect

Once instance starts (1-2 min):
- Click Connect → Open in JupyterLab (easiest — browser based)
- Or use the SSH command shown

### 3.4 — Upload your files

You need to upload:
- output/train_data.jsonl  (~1-1.5GB — the big one)
- train.py
- tagger.py

**Option A — JupyterLab upload button (simplest)**
Click the upload arrow (↑) in the file browser panel.
Select the files. train_data.jsonl takes 5-15 min depending on connection.

**Option B — SCP from Mac terminal (faster)**
```bash
scp -P PORT output/train_data.jsonl root@IP:/root/
scp -P PORT train.py tagger.py root@IP:/root/
```
IP and PORT are shown in the Vast.ai instance details page.

**Option C — Google Drive (best if you'll retrain)**
Upload train_data.jsonl to Google Drive once. On the instance:
```bash
pip install gdown
gdown "https://drive.google.com/uc?id=YOUR_FILE_ID"
```
File ID is the long string in the shareable link URL.

### 3.5 — Set up the instance

In the terminal:
```bash
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install trl datasets transformers accelerate peft bitsandbytes sentencepiece protobuf

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print: NVIDIA A100 80GB PCIe
```

### Go/No-Go Before Renting

Before starting a paid instance, confirm all of the following:

- [ ] Local validation (Phase 1) completed without errors
- [ ] Training data prepared and hash recorded (`shasum -a 256 output/train_data.jsonl`)
- [ ] Base model accessible on Hugging Face (token ready if gated)
- [ ] Prompt template confirmed against model card
- [ ] Disk requirement calculated: base model cache + data + checkpoints + adapter + merge headroom
- [ ] Expected duration estimated and cost ceiling set
- [ ] Download destination on Mac ready
- [ ] `tmux` plan confirmed (training must survive SSH disconnect)

If any item is unchecked, do not rent the instance yet.

### 3.6 — Run training

```bash
# Use tmux so training survives disconnection
tmux new -s training

python train.py --preset a100 --train

# To detach from tmux (training keeps running): Ctrl+B then D
# To reattach later: tmux attach -t training
```

You'll see:
```
Using A100 preset: seq_len=16384, max_words=25000, lora_r=32
Loading model...
Trainable params: 83.9M / 8030M (1.0%)
Starting training...
  Epochs: 2, Batch: 2 (effective: 16)

Step 1/625, Loss: 2.143
Step 20/625, Loss: 1.876
...
Step 625/625, Loss: 0.924
Training complete.
```

**Calibrated time estimates** (based on Windows validation run + A100 scaling):

The Windows 3070 Ti completed a validation step in ~79s at seq_len=256 with 1 example/step.
The A100 80GB is ~8-10x faster per token and can handle seq_len=16384 (64x longer sequences)
with flash attention, meaning far fewer steps are needed to cover each story completely.

| Preset     | seq_len | ~Steps (5k stories) | ~Steps (17k stories) | Est. time (5k) | Est. time (17k) |
|------------|---------|---------------------|----------------------|----------------|-----------------|
| a100       | 16,384  | ~600                | ~1,400               | 5-7 hours      | 10-14 hours     |
| a100_max   | 32,768  | ~600                | ~1,400               | 12-16 hours    | 25-35 hours     |

Cost at $0.55/hr (A100 80 GB typical): a100 ~$3–8, a100_max ~$7–19.

Loss should drop from ~2.0 to ~0.8–1.0. More stories = more steps = longer but better model.
a100 is the recommended starting point — a100_max is for a second run if the first goes well.

If it disconnects: reconnect via SSH and run:
```bash
tmux attach -t training
# or if session gone:
python train.py --preset a100 --train --resume
```

### 3.7 — Download the LoRA weights

Training produces output/lora_model/final/ (~300MB). Download this.

**Via JupyterLab:**
Right-click the lora_model folder → Download (auto-zips it)

**Via SCP from Mac:**
```bash
scp -r -P PORT root@IP:/root/output/lora_model/final/ ./output/lora_weights/
```

### 3.7b — After downloading weights: clean up the instance

Before destroying: double-check you have the lora_model/final/ folder locally.
Check its size — it should be ~250-350MB. If it looks right, destroy the instance.

### 3.7d — If the wayback scraper was cancelled mid-run

The full text is saved to wayback_stories.jsonl incrementally. If you cancelled
before it finished, rebuild the index:
```bash
.venv/bin/python3 wayback_scraper.py --consolidate --output ./output
```
This merges the .jsonl sidecar into wayback_index.json with full text intact.

### 3.8 — STOP THE INSTANCE

Go to Vast.ai → Instances → Destroy (or Stop).
You pay by the hour — destroy it as soon as weights are downloaded.
A 4-hour total session (including setup) costs ~$6–10 at $1.50–2.50/hr.

---

## Phase 4 — Merge and convert (on Windows)

### 4.1 — Copy LoRA weights to Windows

Copy the downloaded lora_weights/ folder to:
C:\MCTraining\output\lora_model\final\

### 4.2 — Merge into full model

```cmd
cd C:\MCTraining
venv\Scripts\activate
python train.py --merge
```

Creates output/merged_model/ (~15GB). Takes 5-10 minutes. Free.

### 4.3 — Convert to GGUF

```cmd
python train.py --gguf
```
This writes convert_to_gguf.py with instructions.
The output is output/gguf_model/model-Q4_K_M.gguf (~4.5GB).

Q4_K_M = best quality/size balance. Loses ~1% quality, runs 2x faster.

---

## Phase 5 — Run on Mac M4

### 5.1 — Copy GGUF to Mac

Copy model-Q4_K_M.gguf (~4.5GB) from Windows to Mac.
Options: USB drive, shared network folder, or:
```bash
# From Mac terminal (both on same network):
scp "WINDOWS_USER@WINDOWS_IP:C:/MCTraining/output/gguf_model/model-Q4_K_M.gguf" ~/Models/
```

### 5.2 — Install LM Studio

Download from https://lmstudio.ai and install.

### 5.3 — Load your model

1. Open LM Studio
2. My Models (left sidebar) → Add model from files
3. Select model-Q4_K_M.gguf
4. The model appears in your library

### 5.4 — Configure for M4

Click the model → Edit configuration:
- Context length: 8192 (keep low — Mac has 16GB unified memory shared with GPU; higher contexts cause memory pressure and slow inference significantly)
- GPU layers: Max (slider all the way right)
- Temperature: 0.85
- Repeat penalty: 1.15
- Top P: 0.92
- Prompt format: ChatML

> **Mac memory note:** The M4 with 16GB unified memory runs the 8B Q4 model fine for
> interactive story generation (short prompts, ~50-60 tok/sec). However, bulk
> inference tasks like arc tagging — which send 5,000–10,000 words per story — cause
> severe memory pressure at this RAM size. The model + OS + background apps fills
> all 16GB, macOS starts compressing memory, and inference drops to ~60 sec/story.
> For bulk arc tagging, use the Windows PC instead (see Phase 6).

### 5.5 — Start the server for the browser

1. Local Server tab (left sidebar)
2. Select your model
3. Start Server (default port 1234)

Open the MC Archive browser:
```bash
make browse
```
Go to Generate — status bar should show green "Connected".
You can now generate stories with your fine-tuned model and full RAG.

---

## Phase 6 — Narrative Arc Tagging (do this after the model is working)

Once your fine-tuned model is running in LM Studio, you can use it to add a
deeper layer of tags that regex and semantic tagging cannot produce — because
they require actually reading and understanding each story's arc.

### What arc tags add

The existing auto_tags cover *how* the control works and *where* the story is
set. Arc tags cover *what happens* — the things readers care about most:

| Tag | Examples |
|-----|---------|
| `ending_state` | fully willing, broken, stepfordized, unaware, rescued, ambiguous |
| `subject_change` | complete transformation, bimboized, drone/blank, no change |
| `power_shift` | dominant gains power, power reversal, stable |
| `twist` | reversal, revelation, the controller controlled, no twist |
| `relationship_type` | romantic, ownership/slave, clinical, harem/collection |
| `emotional_tone_end` | warm/happy, dark/disturbing, hollow/empty, triumphant |
| `story_status` | complete, incomplete, open/ongoing, cliffhanger |

These make filters like "stories where she ends up willing" or "stories with a
twist ending" possible — things that are simply invisible to keyword matching.

### Why use the fine-tuned model for this

A base model can classify arcs, but your fine-tuned model knows the genre's
specific vocabulary — it correctly reads "stepfordized", "bimboized", "drone",
and the particular power-dynamic language as meaningful signals rather than
noise. Classification quality is noticeably better.

### Running arc detection

**Run this on your Windows PC, not the Mac.** The Mac's 16GB unified memory is
fully consumed by the model + OS, causing macOS memory compression and ~60 sec/story.
The Windows RTX 3070 Ti runs the entire model in its 8GB VRAM (separate from system
RAM), giving ~25 sec/story — roughly 3× faster.

**Windows setup:**
1. Install LM Studio on Windows
2. Load your Dolphin 3.0 Llama 3.1 **8B** Q4_K_S model (not 12B — the 32,768-token context needed for arc tagging does not fit with 12B weights in 8GB VRAM; 8B is adequate for classification)
3. Set context length to **32768** (~4.7GB weights + ~1GB KV cache = ~5.7GB total — fits the 8B comfortably)
4. Start the server on port 1234
5. Run from the Dropbox-synced folder:

```bash
# Test on 20 stories first — check the output looks sensible
make arc-test

# Full run — all 17k stories (~120 hours at 25 sec/story; run in batches overnight)
python narrative_arc.py --limit 2000

# If it was interrupted, resume automatically (already-tagged stories are skipped)
python narrative_arc.py --limit 2000

# Check coverage when done
make arc-stats
```

The index saves every 5 stories — safe to Ctrl+C and resume anytime. Results
sync back to the Mac via Dropbox automatically.

**Mac fallback:** If you must run on Mac, set LM Studio context to 16384 and
`MAX_STORY_WORDS = 5000` in narrative_arc.py. Expect ~60 sec/story and close
all Chrome tabs and Jupyter notebooks first to minimize memory pressure.

### After arc detection

```bash
# Rebuild everything with the new tags
make post-update
```

This re-exports JSONL (training data now includes arc tags) and rebuilds the
recommendation cache (recs now factor in arc preferences — liked stories with
"fully willing" endings will push similar arc outcomes higher).

### Optional: retrain with arc tags

If you want a model that can write toward specific arc outcomes (e.g. "write a
story that ends with stepfordization"), retrain after arc detection:

```bash
# On Mac — re-prepare training data (now includes arc tags in prompts)
python train.py --preset a100 --chunk --prepare

# Upload new train_data.jsonl to A100 and retrain as before
```

The arc tags are included in the training prompt template automatically.
Stories with `ending_state: fully willing` will be paired with prompts that
include that tag, so the model learns to write toward it on request.

---

## Hardware Notes

### What runs where

| Task | Mac M4 (16GB) | Windows RTX 3070 Ti (32GB + 8GB VRAM) |
|------|--------------|----------------------------------------|
| Scraping / data prep | ✅ Fine | ✅ Fine |
| SetFit classifier training | ✅ ~3h on MPS | ✅ ~25 min on CUDA |
| SillyTavern — 8B inference (LAN server) | ✅ ~17 tok/s (Metal) | ✅ ~45–60 tok/s (CUDA) |
| SillyTavern — 12B inference (LAN server) | ⚠️ ~10–15 tok/s, memory pressure | ✅ ~30–40 tok/s, Q4_K_S required |
| Story generation (interactive) | ✅ ~50-60 tok/sec | ✅ Fast |
| Arc tagging (bulk, 17k stories) | ⚠️ ~60 sec/story, memory pressure | ✅ ~25 sec/story, full VRAM |
| Chat LoRA fine-tune (Pipeline B) | ❌ Too slow | ✅ ~8–10 hours (12B, tight — prefer A100); ~5 hours (8B fallback) |
| Story LoRA fine-tune (Pipeline A) | ❌ Too slow | ⚠️ Validates pipeline only — use A100 |

### Why arc tagging is slow on Mac

The 8B Q4_K_S model needs ~4.7GB. The Mac has 16GB unified memory shared between
CPU and GPU. With the OS, scraper, browser, and other apps running, the system
fills all 16GB and starts compressing memory — inference drops from ~50 tok/sec
to ~15 tok/sec or worse. Closing everything helps but doesn't fully solve it.

> **Note on Mac speed figures:** Two different figures appear in this guide — ~17 tok/s and ~50–60 tok/s. These are not contradictory: ~17 tok/s is **[Measured]** generation speed in LM Studio with GGUF format under memory pressure. ~50–60 tok/s refers to **[Estimated]** MLX-format throughput on M-series chips, which uses unified memory more efficiently. For GGUF inference (the workflow in this guide), 17 tok/s is the correct planning figure. For MLX models, 50–60 tok/s is plausible but depends on quantization and model architecture.

### Windows RTX 3070 Ti setup for arc tagging

- Load model in LM Studio with **context 32768** (fits: ~4.7GB weights + ~1GB KV cache = 5.7GB, well within 8GB VRAM)
- The model runs entirely in VRAM — system RAM is free for everything else
- Set `MAX_STORY_WORDS = 10000` in narrative_arc.py (default)
- Expected: ~25 sec/story → 2,000 stories overnight (~14 hours)

### Mac LM Studio settings (for interactive use only)

- Context length: **8192** — enough for generation, avoids memory pressure
- Do not use Mac for bulk arc tagging runs

---

## LAN Hosting — Windows PC serves the model, Mac runs SillyTavern

You can run the LLM entirely on the RTX 3070 Ti (Windows PC) and connect to it
from SillyTavern on the MacBook over your local network. The Mac acts as the
frontend only — it never loads the model.

### Why do this

The 3070 Ti is significantly faster than the M4 for inference because its VRAM
(dedicated, 608 GB/s bandwidth) is not shared with the OS, browser, or any other
app. The Mac frees up all its RAM for other tasks.

**Measured and estimated performance:**

| Model | Setup | Generation | Prompt eval (5k tokens) | Total (25 msgs) |
|---|---|---|---|---|
| 12B Q4_K_S | RTX 3070 Ti CUDA | ~30–40 tok/s | ~200–300 tok/s | ~22–32 sec |
| 8B Q4_K_M | RTX 3070 Ti CUDA | ~45–60 tok/s | ~350–500 tok/s | ~12–16 sec |
| 8B Q4_K_M | Mac M4 Metal | ~17 tok/s | ~174 tok/s | ~40 sec |

12B on 3070 Ti is the recommended setup for RP — 22–32 sec per turn is comfortable
for interactive sessions. 8B numbers are real measured values; 12B numbers are
estimated from GPU scaling. Use 8B only as a hardware fallback if 12B Q4_K_S
consistently causes OOM.

### Setup

**On the Windows PC:**

1. Install LM Studio (or KoboldCPP — see below)
2. Load your model (Q5_K_M recommended for 3070 Ti — fits in 8GB, better quality than Q4)
3. **LM Studio settings:**
   - Parallel Slots: **1** (you're the only user — multiple slots waste KV cache memory)
   - Context: **8192**
   - GPU layers: all (slider fully right)
   - Flash Attention: on (if available)
4. Start the server on port **1234** (LM Studio default)
5. Allow the port through Windows Firewall:
   ```
   Windows Defender Firewall → Advanced Settings
   → Inbound Rules → New Rule → Port → TCP 1234 → Allow
   ```

> **Security note:** The model server runs plain HTTP on your LAN — prompts and generated text are unencrypted over the network. Scope the Windows Firewall rule to your LAN subnet or the Mac's specific IP rather than all networks. Do not forward this port through your router. Do not expose it to the public internet. The content passing through this connection is sensitive. For remote access (away from home), use Tailscale or another VPN rather than port forwarding.

6. Find your PC's local IP:
   ```cmd
   ipconfig
   ```
   Look for `IPv4 Address` under your Wi-Fi or Ethernet adapter, e.g. `192.168.1.45`

**On the Mac (SillyTavern):**

1. In SillyTavern → plug icon → Chat Completion → Custom (OpenAI-compatible)
2. URL: `http://192.168.1.45:1234/v1` (use your PC's actual IP, not localhost)
3. Click Connect — you should see a green checkmark
4. Select your model from the dropdown

**Alternative: KoboldCPP (slightly faster CUDA, no GUI)**

KoboldCPP is a more optimised llama.cpp build that typically runs 5–10% faster
than LM Studio on CUDA. It also handles the firewall more automatically.

```
Download koboldcpp.exe from github.com/LostRuins/koboldcpp/releases
Run it → Browse → select GGUF → GPU Layers: 40 (all, for 12B Nemo/Magnum) → Port: 5001 → Launch
```

SillyTavern connection: API type → **KoboldAI** → URL: `http://192.168.1.45:5001`

### Optimal 3070 Ti settings

**12B model — recommended (e.g. Dolphin Nemo 12B Q4_K_S or Magnum 12B Q4_K_S):**

| Setting | Value | Notes |
|---|---|---|
| Model | Q4_K_S (~6.5GB) | Must use K_S not K_M — K_M overflows with KV cache |
| Context | 6144 | ~1.0GB KV cache; total ~7.5GB, safe |
| GPU layers | All (40 for 12B) | Full VRAM — no CPU offload |
| Parallel slots | 1 | Single user |
| Flash attention | On | More important at 12B to reduce KV memory |
| Prompt format | Match model | Mistral Nemo / Magnum uses its own template — check model card |

On the Mac SillyTavern side, set Max Context to **5500** when using 12B to leave
generation headroom within the 6144 server limit.

**8B model — fallback only (e.g. Sunfall Q5_K_M):**

| Setting | Value | Notes |
|---|---|---|
| Model | Q5_K_M (~5.7GB) | Fits with headroom for KV cache; better quality than Q4 |
| Context | 8192 | ~1.3GB KV cache; total ~7.0GB, comfortable |
| GPU layers | All (33 for 8B) | Full VRAM — no CPU offload |
| Parallel slots | 1 | Single user, eliminates KV cache eviction |
| Flash attention | On | Speeds up long contexts |
| Prompt format | Llama 3 | Sunfall uses Llama-3 chat template |

**VRAM budget for 12B on 8GB — why Q4_K_S is required:**
```
Q4_K_M weights:  ~7.0GB
KV cache 6144:   ~1.0GB
Total:           ~8.0GB  ← exactly at limit, unstable

Q4_K_S weights:  ~6.5GB
KV cache 6144:   ~1.0GB
Total:           ~7.5GB  ← 500MB headroom, stable
```

### What "KV cache eviction" means and why it slows you down

When parallel slots is set to 4 (LM Studio default), the KV cache is split four
ways. When one slot fills up, LM Studio forcibly evicts another slot's entire
context — visible in the logs as:

```
decode: failed to find a memory slot for batch of size 512
srv  try_clear_id: purging slot 0 with 7807 tokens
```

That evicted slot then has to re-process its full conversation from scratch on the
next request, adding 30–60 seconds. Setting parallel slots to 1 eliminates this.

### When the Mac is faster

The Mac is preferable for generation when:
- The model is MLX-format (Mac-native) rather than GGUF — MLX on M4 matches
  or exceeds llama.cpp CUDA on the 3070 Ti for some workloads
- You need portability (laptop, away from home network)
- The Windows PC is busy with another task (training, arc tagging)

For SillyTavern RP at home: default to the 3070 Ti. For travel or when the PC
is training: fall back to the Mac.

---

## Troubleshooting

**Windows: CUDA out of memory**
```cmd
python train.py --train --preset 3070 --seq-len 1024
```

**Windows: unsloth install fails**
```cmd
pip install unsloth
pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

> **[Important — read before local validation]** The Windows data-collator workaround (below) causes a significant portion of each training example to be silently dropped during local validation. This means local validation does NOT validate that your training data is handled correctly at full sequence length. It only confirms the training loop runs. Always run a real A100 pilot before spending significant money on a full training run.

**Windows: "Input IDs of shape [1, 1024] > max sequence length" warning during training**
This is a known unsloth issue on Windows. The data collator defaults to max_length=1024
but the model only processes --seq-len tokens (e.g. 256). Unsloth truncates input_ids
but not labels, requiring a patch to llama.py. The patched venv in C:\MCTraining handles
this automatically (labels are truncated to match hidden_states in _CausalLM_fast_forward).
Effect: 75% of each training example is silently dropped when seq-len=256 with the 1024
collator default. For validation runs this is fine. For real training on the A100, this
issue does not exist (Linux unsloth works correctly without any patches).

**Vast.ai: disk full**
```bash
df -h
du -sh ~/.cache/huggingface/  # model cache (~5GB)
# Rent instance with more disk, or clear cache between runs
```

**LM Studio: generates gibberish**
Switch Prompt Format to ChatML — dolphin uses ChatML format.

**Loss never drops below 1.5**
Lower learning rate: add --learning-rate 1e-4 to train command.

**Loss drops too fast (below 0.5 by epoch 1)**
Overfitting on too little data. Add more stories or reduce epochs to 1.

> **Note:** Loss scale depends on tokenizer, masking strategy, sequence packing, and corpus entropy — universal thresholds do not transfer between runs. Use trends instead: finite and decreasing training loss, stable validation loss, no widening train-validation gap. The figures above are observations from a specific run configuration, not universal targets.

---

## File sizes and costs

| File | Size | Location |
|------|------|----------|
| stories_index.json | ~500MB | Mac → copy to Windows |
| train_data.jsonl | ~3–5GB | Mac → upload to A100 |
| lora_model/final/ | ~300MB | Download from A100 |
| merged_model/ | ~15GB | Windows (can delete after GGUF) |
| model-Q4_K_M.gguf | ~4.5GB | Copy to Mac |

| Cost | Amount |
|------|--------|
| Vast.ai A100 (~4–6hr total incl. setup) | ~$2–4 |
| Vast.ai bandwidth (upload ~4GB) | ~$0–2 |
| **Total** | **~$2–6** |

Stop the instance as soon as LoRA weights are downloaded.
Shorter runs: use `--preset a100` without `--chunk` for ~$0.70–2.10 (fewer stories, faster).
Longer runs: use `--preset a100_max --chunk` for best quality at ~$1.80–4.20.

---

## Volatile Pricing Snapshot

**Last checked: 2026-07-13 [Volatile]** — GPU rental markets change frequently. Verify current rates at vast.ai before budgeting.

### Vast.ai rental rates (approximate, unverified)

| GPU | VRAM | Approx. rate | Suitable for |
|---|---|---|---|
| A100 40 GB | 40 GB | ~$1.20–1.80/hr | 12B training at ≤16K context |
| A100 80 GB | 80 GB | ~$1.80–2.80/hr | 32B training; 12B at 32K context |
| RTX 3090 | 24 GB | ~$0.35–0.60/hr | 32B inference; not ideal for training |
| RTX 4090 | 24 GB | ~$0.50–0.80/hr | 32B inference, faster than 3090 |

**Total billable cost = hourly rate × (upload + setup + training + merge + download) hours**

Do not estimate only training time. A100 instances charge from the moment they start. A 3-hour training run may cost 4–5 hours of billed time including setup, data upload, and download.

Storage charges apply to stopped (not destroyed) instances. If you pause a run, check whether storage is billed separately.

> Elsewhere in this guide, cost estimates cite this section rather than repeating figures.

---

## 12B Model Option

### 12B is the minimum viable for RP — 8B for hardware fallback or short stories only

For interactive RP with progressive mechanics, 8B does not meet the bar. The emotional
gradation gap (binary resist/yield vs graduated response), the coherence window (~20
messages vs ~40), and the trigger deepening limitations mean an 8B fine-tune falls below
OurDream base tier for sustained RP — the exact use case this guide targets. 12B clears
that bar and matches OurDream with memory augmentation. Train 12B first; only drop to 8B
if the 3070 Ti consistently OOMs during 12B training.

### What actually improves at 12B

The jump from 8B → 12B is meaningful across every RP-relevant dimension:

**For story generation:**

| Dimension | 8B | 12B | Notes |
|---|---|---|---|
| Sentence variety | Moderate | Good | Less structural repetition over long passages |
| Character voice | Tends to homogenise | Stays distinct longer | Different characters sound more different |
| Coherence window | Degrades ~2,000 words | Holds ~3,500 words | Plot threads and names stay consistent further |
| Emotional nuance | Shallow; tells > shows | Noticeably better | Internal states feel more earned |
| Tag adherence | Excellent | Excellent | Fine-tuning fixes this for both sizes |
| Genre vocabulary | Excellent (fine-tuned) | Excellent (fine-tuned) | No meaningful difference |

The 8B model is good for short stories (under ~2,000 words) and scene drafts.
The 12B model is meaningfully better for anything longer — the extra coherence
window is the most practically important improvement.

**For SillyTavern chatbot RP (Pipeline B):**

The 12B improvements show up differently in short-turn conversation than in long-form prose:

| Dimension | 8B | 12B | How it shows |
|---|---|---|---|
| Per-turn response | Good | Better | More natural phrasing, less repetitive sentence structures |
| Session consistency | Drifts after ~20 messages | Holds longer | Remembers established triggers, tone, and dynamics |
| Emotional gradation | Binary (resisting / yielding) | Graduated | Can hold partial compliance, slow yielding, conflicted states |
| Reactivity precision | Responds to the general idea | Responds to specific words | Picks up on subtle framing in your messages |
| Character voice | Consistent per session | More consistent | The subject's personality stays coherent over long sessions |

The coherence window advantage is especially important for RP — a 12B subject
will remember that a specific trigger was established 30 messages ago and react
consistently without you having to repeat it.

The real quality ceiling — genuine emotional depth, originality — remains out of
reach at both sizes. That gap requires 70B+ and is not closable locally.

### Hardware implications

**Training (A100 rental):**
- ~40% more time than the 8B equivalent preset
- ~40% more cost (~$9–22 for `a100_12b` vs ~$6–15 for `a100`)
- No other changes needed — A100 80GB handles 12B comfortably

**Chat LoRA training on RTX 3070 Ti (Pipeline B):**
- 12B with QLoRA at seq_len=2048: ~7–7.5GB total — tight but usually fits
- If OOM: add `--lora-r 4` to reduce LoRA memory footprint
- Expect ~8–10 hours for full 161k conversation dataset
- Safer to use A100 for 12B chat training (~$1–2, ~1.5 hours)

**Inference on RTX 3070 Ti as LAN server (SillyTavern):**
- Use Q4_K_S (~6.5GB) — Q4_K_M overflows with KV cache at conversational context
- Set context to **6144** (not 8192) — leaves headroom for KV cache
- GPU layers: all 40 layers in VRAM
- Generation speed: ~30–40 tok/s (vs ~45–60 tok/s for 8B) — still comfortable for RP
- Total turn time (25-message conversation): ~22–32 sec vs ~12–16 sec for 8B
- Arc tagging: reduce `MAX_STORY_WORDS` to 7,000 to keep within VRAM

**Local inference on Mac M4 (16GB unified):**
- Q4_K_M runs fine — 7GB weights + OS + apps fits comfortably in 16GB
- No changes needed vs the 8B workflow
- Generation speed ~10–15 tok/s (slower than 8B but usable for travel/fallback)

### Training commands

```bash
# Prepare data (same as 8B — data format is model-agnostic)
python train.py --preset a100_12b --chunk --prepare

# Train on A100 — specify the 12B base model
python train.py --preset a100_12b \
  --model cognitivecomputations/dolphin3.0-mistral-nemo-12b \
  --train

# Maximum context variant (best quality, ~2× longer to train)
python train.py --preset a100_max_12b \
  --model cognitivecomputations/dolphin3.0-mistral-nemo-12b \
  --train
```

After training, the merge and GGUF conversion steps are identical to the 8B
workflow. When converting, use Q4_K_S for the 3070 Ti or Q4_K_M for Mac-only use.

### Fine-tuning stack comparison (8B vs 12B vs 32B, all combinations)

The base model is the frozen foundation your LoRA builds on — a stronger RP base
produces a better result from the same training data. "Base" below means a raw
instruction-tuned model with no RP fine-tuning; named models (Sunfall, Dolphin, Magnum)
are already RP/prose optimised before your LoRA is applied.

| Stack | Genre depth | Subject voice | Prose | Content | Training cost |
|---|---|---|---|---|---|
| Base 8B + Story + Chat | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | A100 + 3070 Ti |
| Sunfall 8B + Chat only | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★★★ | 3070 Ti only (free) |
| Sunfall 8B + Story + Chat | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★★ | A100 + 3070 Ti |
| Base 12B + Story + Chat | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | A100 × 2 |
| Dolphin Nemo 12B + Chat only | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★★ | A100 or 3070 Ti (tight) |
| Dolphin Nemo 12B + Story + Chat | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | A100 × 2 |
| Magnum 12B + Story + Chat | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | A100 × 2 |
| Magnum 32B + Story + Chat | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | A100 × 2 (longer) |

Key points:
- **Dolphin Nemo 12B + Chat only** is the entry point — minimum viable for RP, ~$1–2 on A100
- **Sunfall 8B + Chat only** is the hardware-constrained fallback — free on 3070 Ti, not recommended for RP
- **Dolphin Nemo 12B + Story + Chat** is the 12B quality ceiling at similar prose to Magnum 12B
- **Magnum 32B + Story + Chat** is the overall quality ceiling — strongest prose base, largest coherence window, best character voice from the frozen weights before your LoRA even starts
- Using a raw "Base" model vs a named RP model costs you ~1 star on prose and subject voice for the same training run — always prefer a named RP base

### Recommended path

1. **Start here (~$1–2):** Dolphin Nemo 12B + Chat only — train on A100 (~1.5 hours), load Q4_K_S on 3070 Ti at ctx 6144. Minimum viable RP setup.
2. **Full 12B quality (~$4–11 total):** Add the story fine-tune on A100 for genre depth. Magnum 12B + both pipelines is the 12B ceiling.
3. **Quality ceiling (~$10–25 total + 3090 GPU):** Magnum 32B + both fine-tunes — see the 32B Model Option section below.
4. **Optional:** `--preset a100_max` after arc tagging is complete so arc tags are included in training prompts and the model learns to write toward specific arc outcomes.
5. **Hardware-constrained fallback (free, 3070 Ti only):** Sunfall 8B + Chat only — acceptable for short story generation under ~2,000 words. Not recommended for progressive RP mechanics.

---

## 32B Model Option

### Is it worth training a 32B instead of 12B?

**Yes, with a hardware caveat.** The quality jump from 12B → 32B is larger than 8B → 12B,
particularly for RP coherence and long-form prose. A 32B model requires ~19–20 GB of VRAM
to load at Q4 quality — more than the RTX 3070 Ti (8 GB) or Mac M4 (16 GB unified) can hold.
You would need a GPU upgrade or to rent inference on Vast.ai for every session.

### What actually improves at 32B

**For story generation:**

| Dimension | 8B fine-tuned | 12B fine-tuned | 32B fine-tuned |
|---|---|---|---|
| Prose quality | ★★★☆☆ | ★★★½☆ | ★★★★☆ |
| Coherence window | ~1,500 words | ~3,500 words | ~7,000 words |
| Sentence variety | Flattens after ~500 words | Good to ~1,500 words | Strong to ~4,000 words |
| Distinct character voices | Holds ~4–5 chars | Holds ~8–10 chars | Holds ~15+ chars |
| Arc resolution pacing | Tends to rush | Better | Varied and deliberate |
| Shows vs tells emotion | Mostly tells | Mixed | Mostly shows |
| Genre accuracy | ★★★★★ | ★★★★★ | ★★★★★ |
| Tag / prompt adherence | ★★★★★ | ★★★★★ | ★★★★★ |

**For chatbot / SillyTavern RP:**

| Dimension | 8B fine-tuned | 12B fine-tuned | 32B fine-tuned |
|---|---|---|---|
| Messages before character drift | ~20 | ~40 | ~80 |
| Remembers established triggers | ~15 messages | ~35 messages | ~75 messages |
| Emotional gradation | Binary (resisting / yielding) | Graduated | Nuanced and sustained |
| Reactivity to subtle framing | Responds to general idea | Responds to specific words | Responds to implication |
| Sustained internal state | Resets often | Moderate continuity | Strong continuity |
| Response naturalness | Good | Very good | Excellent |

The coherence and drift numbers are community-observed estimates, not hard guarantees — actual
results depend on base model, quantization, and SillyTavern context settings.

### Hardware implications

**Training:**

| Hardware | Story LoRA (Pipeline A, 16k seq) | Chat LoRA (Pipeline B, 2k seq) |
|---|---|---|
| RTX 3070 Ti (8 GB) | ❌ | ❌ (too large at any seq_len) |
| A100 40 GB | ⚠️ fits at seq_len ≤ 8,192 | ✅ fits at seq_len=2,048 |
| A100 80 GB | ✅ recommended | ✅ comfortable |

Both pipelines need A100 time for 32B — the 3070 Ti cannot train either.

**Inference:**

| Hardware | 32B Q4_K_M | Speed | Notes |
|---|---|---|---|
| RTX 3070 Ti (8 GB) | ❌ | — | Cannot load model |
| Mac M4 16 GB unified | ❌ | — | ~3–4 GB short even at minimal quantization |
| RTX 3090 (24 GB) | ✅ | ~25–35 tok/s | 4 GB KV headroom; keep ctx ≤ 8,192 |
| RTX 4090 (24 GB) | ✅ | ~40–55 tok/s | Same VRAM as 3090 but much faster |
| Vast.ai 3090 rental | ✅ | ~25–35 tok/s | $0.13–0.25/hr; ~2–3 min cold start per session |

### Training costs

At current Vast.ai pricing ($0.45–0.70/hr for A100 80 GB):

| Pipeline | Est. time | Est. cost |
|---|---|---|
| Story LoRA (Pipeline A, `a100` preset) | 8–18 hours | $3.60–12.60 |
| Chat LoRA (Pipeline B, 50k conversations, A100 40 GB) | 3–5 hours | $0.90–2.25 |
| Chat LoRA (Pipeline B, full 161k, A100 40 GB) | 12–18 hours | $3.48–8.10 |
| Both pipelines (story + 50k chat) | 11–23 hours | $4.95–16.10 |

Compare to 8B (~$1–3 total) and 12B (~$3–7 total) — the 32B premium is real but still a
one-time expense, after which inference is free if you own suitable hardware.

### Inference costs (monthly at ~43 hours/month)

**Option A — Vast.ai rental for every session:**

| GPU | Rate | Monthly cost (43 hrs) | Fits $20 budget? |
|---|---|---|---|
| RTX 3090 (budget listing) | $0.13–0.18/hr | $5.59–7.74 | ✅ comfortably |
| RTX 3090 (typical listing) | $0.18–0.25/hr | $7.74–10.75 | ✅ yes |
| RTX 4090 | $0.35–0.50/hr | $15.05–21.50 | ⚠️ borderline |

A 3090 at typical rates leaves $9–14/month for A100 training bursts. Friction: each
session requires the instance running and model pre-loaded (~2–3 min cold start from stopped).

**Option B — Buy a used RTX 3090 (~$300–500 secondhand):**

Free inference after purchase. Runs 32B Q4_K_M at ~25–35 tok/s at ctx=8,192.
Break-even vs Vast.ai at $0.20/hr average: ~$400 / ($0.20 × 43 hrs) ≈ 47 months —
but eliminates session startup friction entirely and frees your $20/month budget for A100 training.

### Training commands

```bash
# Data preparation is the same as 8B/12B — format is model-agnostic
python train.py --preset a100 --chunk --prepare

# Story fine-tune on A100 80 GB
python train.py --preset a100 \
  --model anthracite-org/magnum-v4-32b \
  --train

# Chat LoRA on A100 40 GB (cannot run on 3070 Ti at 32B)
python train_chat.py --train --preset a100 \
  --model anthracite-org/magnum-v4-32b
```

**Recommended 32B base:** `anthracite-org/magnum-v4-32b` (Magnum by Anthracite) —
purpose-built for RP and creative writing, designed explicitly to be fine-tuned further.
Prose variety, character voice, and fiction-writing behaviour are already established in
the frozen weights before your LoRA starts. This means your genre LoRA only needs to teach
the MC erotica delta, not general fiction craft on top of an instruction model.

**Fallback:** `Qwen/Qwen2.5-32B-Instruct` — strong instruction following and fine-tunes
well with QLoRA, but weaker RP defaults than Magnum. Use if Magnum is unavailable or if
a newer Anthracite release is out. Check r/SillyTavernAI and r/LocalLLaMA before your
run — a newer Magnum version or community-favourite 32B RP base may have appeared.

### Arc tagging at 32B

If you switch to a 32B model for arc tagging, reduce `MAX_STORY_WORDS` to 6,000 in
`narrative_arc.py` to stay within VRAM on a rented 3090, and expect ~40–60 sec/story
(vs ~25 sec for 8B on the 3070 Ti).

---

## Model Quality — Local vs Cloud

### What your fine-tuned models actually produce

**12B fine-tuned — recommended target:**

Fine-tuning Mistral Nemo 12B or Magnum 12B on your corpus gives a model that matches
OurDream base tier on prose quality and exceeds it on genre specificity.

**Strengths:**
- Stays on-genre natively — knows the MC/hypnosis vocabulary, tropes, and power
  dynamic language without prompting
- Follows tag prompts accurately (mc_type, setting, themes, dominance, etc.)
- Graduated emotional responses — can hold partial compliance, conflicted states, slow yielding
- Coherence holds to ~40 messages without augmentation, ~80–100 with full memory setup
- Handles progressive mechanics (Training Log, stage tracking) with genuine nuance
- Integrates with your story bibles and the archive browser seamlessly
- Runs entirely locally — zero cost per generation, no API limits, fully private

**Weaknesses:**
- Emotional depth ceiling — real interiority and genuine literary surprise remain out of reach
- Prose variety can still flatten over very long passages
- Tight on 3070 Ti — Q4_K_S at ctx 6144 only; context window narrower than 8B

**Honest tier:** Matches a competent RP-tuned commercial model. Competitive with OurDream
base tier for sustained RP. Below 32B on prose depth but achievable on your current hardware.

---

**8B fine-tuned — hardware-constrained fallback:**

Fine-tuning Llama 3.1 8B improves genre adherence significantly but does not meet the bar
for progressive RP mechanics. Use only if 12B training consistently OOMs on your 3070 Ti.

**Where it holds up:**
- Short story generation under ~2,000 words — genre vocabulary and tag adherence are excellent
- Scene drafts and premise generation — fast iteration, not sustained sessions

**Where it falls short for RP:**
- Emotional gradation is binary (resisting / yielding) — cannot hold graduated states
- Session coherence drifts after ~20 messages even with full augmentation
- Trigger deepening resets or goes formulaic; progressive mechanics flatten
- Falls below OurDream base tier for sustained RP

**Honest tier:** Competent short-form genre writer. Not viable as a primary RP model.

---

### Quality comparison table

A typical paid companion app (~$12–20/mo) is used as the commercial baseline throughout.
OurDream.ai is cited as a concrete example, but the ratings below are **estimates** —
OurDream does not disclose what models they run. Based on their published tier names and
descriptions, the picture is roughly:

| OurDream tier | Cost | What it likely is |
|---|---|---|
| Standard / Balanced / Lively / Muse | Included in subscription | 7–20B models; **auto-summary included**; optional pinned memories |
| Genius / Lively Genius / Muse Genius | 1 coin/message | Larger model; **"improved memory"** explicitly named — likely enhanced summarisation or RAG |
| Mastermind | 1 coin/message | Probably their largest or frontier-API model |

The base subscription includes platform-level auto-summarisation and optional pinned memories.
"Improved memory" at Genius tier is likely a step up from this (better summarisation,
or retrieval-augmented), not the introduction of memory from scratch. The comparison
below represents the base subscription tiers (no coins), which is the most common usage.

What companion apps provide at base tier: image/video/voice generation, polished UX,
cross-session character persistence, auto-summary, and optional pinned memories — no setup
required. What they do not provide: fine-tuning on your corpus, user-controlled lorebooks,
or SillyTavern integration. The ST equivalent of OurDream's base tier is ST **with**
vector storage or summary injection configured, not bare ST.

| Option | Prose quality | Genre accuracy | One-time training | Monthly inference | Privacy |
|--------|--------------|----------------|-------------------|-------------------|---------|
| **Companion app† (mid tier)** | ★★★☆☆ | ★★☆☆☆ | $0 | ~$12–20 | ❌ |
| **Your fine-tuned 8B** | ★★★☆☆ | ★★★★★ | ~$1–3 | Free (3070 Ti) | ✅ |
| **Your fine-tuned 12B** | ★★★½☆ | ★★★★★ | ~$3–7 | Free (3070 Ti) | ✅ |
| **Your fine-tuned 32B** | ★★★★☆ | ★★★★★ | ~$5–16 | Free (own 3090) or ~$6–11/mo | ✅ |
| **NovelAI Clio (~8B)** | ★★★☆☆ | ★★★★☆ | $0 | ~$10/mo | ❌ |
| **NovelAI Kayra (~13B)** | ★★★★☆ | ★★★★☆ | $0 | ~$15/mo | ❌ |
| **Venice.ai (Llama 70B)** | ★★★★☆ | ★★★☆☆ | $0 | ~$10–15/mo | ❌ |
| **Mancer / Featherless** | ★★★☆☆ | ★★★☆☆ | $0 | Pay-per-token | ❌ |
| **Skilled human author** | ★★★★★ | ★★★★★ | N/A | N/A | ✅ |

†Companion app ratings are estimates. The underlying model is typically undisclosed;
cheaper tiers likely route to smaller or quantised models. Treat these as a lower-end
baseline for the category, not a measured benchmark for any specific service.

‡8B fine-tuned falls below the minimum viable threshold for sustained RP with progressive
mechanics. The emotional gradation and session coherence figures reflect a ceiling the model
cannot exceed regardless of augmentation. 12B is the minimum recommended size for this use case.

**Genre accuracy note:** Local fine-tuned models score ★★★★★ because your 17,000-story
corpus is baked in — the model knows the tag vocabulary, MC genre conventions, and
power-dynamic language. Companion apps and cloud services start from ★★☆☆☆ because they
know nothing about your specific corpus. Genre fine-tunes like NovelAI's sit between
those at ★★★★☆ — trained on fiction broadly, but not on your archive.

**Not viable for this use case (content policy):**
OpenAI (GPT-4), Anthropic (Claude), Google (Gemini), and Mistral's API all refuse
explicit content. No prompting workaround reliably bypasses this. Do not build a
workflow that depends on them for generation.

---

### Story generation performance

| | Companion app† (base tier) | 8B fine-tuned | 12B fine-tuned | 32B fine-tuned | NovelAI Kayra |
|---|---|---|---|---|---|
| **Prose quality** | ★★★☆☆ | ★★★☆☆ | ★★★½☆ | ★★★★☆ | ★★★★☆ |
| **Coherence window** | ~2,500–4,000 words | ~1,500 words | ~3,500 words | ~7,000 words | ~4,000 words |
| **Sentence variety** | Moderate (some repeated patterns) | Flattens ~500 words | Good to ~1,500 | Strong to ~4,000 | Good |
| **Distinct character voices** | 4–6 | 4–5 | 8–10 | 15+ | 6–8 |
| **Arc pacing** | Tends to rush | Tends to rush | Better | Deliberate | Good |
| **Shows vs tells emotion** | Mixed (mostly tells, structural aids help) | Mostly tells | Mixed | Mostly shows | Mixed |
| **Genre accuracy** | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★☆ |
| **Tag / prompt adherence** | ★☆☆☆☆ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★☆☆ |
| **5,000+ word coherence** | ★★★☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★★☆☆ |
| **Monthly cost** | ~$12–20 | Free | Free | Free or ~$6–11* | ~$15 |

*Free if you own an RTX 3090; ~$6–11/mo renting a Vast.ai 3090 at ~43 hrs/month.

†Base/mid tier (no per-message coins) — "improved memory" is a paid coin feature on OurDream. Companion app coherence figures updated based on analysis of actual session samples; prior estimate of ~800 words was too conservative for mid-tier models.
Coherence window = approximate word count before plot threads, names, or timeline start to drift noticeably.

---

### Chatbot / SillyTavern RP performance

| | Companion app† | 8B fine-tuned | 12B fine-tuned | 32B fine-tuned |
|---|---|---|---|---|
| **Messages before character drift** | ~50–80 (no card aids‡) / 100+ (designed card) | ~20 | ~40 | ~80 |
| **Remembers established triggers** | ~40–60 messages | ~15 messages | ~35 messages | ~75 messages |
| **Emotional gradation** | Moderate (with good card design) | Binary (resist/yield) | Graduated | Nuanced, sustained |
| **Reactivity to subtle framing** | Moderate | General idea | Specific words | Responds to implication |
| **Sustained internal state** | Moderate (boosted by character card structure) | Resets often | Moderate continuity | Strong continuity |
| **First-person consistency** | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★★★★ |
| **Genre / dynamic vocabulary** | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★★★★ |
| **Image / voice generation** | ✅ (core feature) | ❌ | ❌ | ❌ |
| **SillyTavern lorebook support** | ❌ | ✅ | ✅ | ✅ |
| **Monthly cost** | ~$12–20 | Free | Free | Free or ~$6–11/mo |

‡Companion app coherence estimates calibrated against actual session samples (100+ message
sessions, multiple character types). OurDream base tier includes platform auto-summarisation
and optional pinned memories — the observed coherence is a combined effect of model + these
infrastructure features, not raw context window alone. Even so, well-designed character cards
(in-context state summaries, structured inner-monologue blocks) push coherence further.
The ST equivalent requires memory augmentation to be configured (vector storage, summary
injection, lorebook) — bare ST without those will underperform OurDream base tier on
long-session coherence. "Improved memory" coin tier likely adds enhanced summarisation
or retrieval on top of the base auto-summary.

The session coherence numbers are the most practically relevant for RP: a 32B model edges
ahead on raw coherence and significantly ahead on prose depth and emotional nuance, but
the companion app mid-tier is no longer the easy win the prior numbers implied.
The clear local advantage is **genre specificity**: your 17k-story corpus and tag vocabulary
beat any companion app from message one, and that gap widens in scenes requiring precise
power-dynamic language the app's model has never seen in training.

---

### Cost breakdown — training and running

| | 8B (both pipelines) | 12B (both pipelines) | 32B (both pipelines) | Companion app† |
|---|---|---|---|---|
| **Training (one-time)** | ~$1–5 | ~$3–9 | ~$5–16 | $0 |
| **Inference hardware needed** | 3070 Ti ✅ (you own it) | 3070 Ti ✅ | RTX 3090+ required | None |
| **Monthly running cost** | Free | Free | Free (own 3090) or ~$6–11 | $19.99 |
| **Year 1 total (Vast.ai inference)** | ~$1–5 | ~$3–9 | ~$77–148 | ~$144–240 |
| **Year 1 total (buy used 3090, ~$400)** | ~$1–5 | ~$3–9 | ~$405–420 | ~$144–240 |
| **Year 2+ annual cost** | $0 | $0 | $0 (own 3090) or ~$72–132 (Vast.ai) | ~$144–240 |

†Companion app year costs assume $12–20/mo; cheaper tiers may use lower-quality models.
Year 1 on Vast.ai for 32B inference (~$77–148) is cheaper than even a discounted companion
app subscription and significantly more capable for genre-specific writing and RP.
Break-even on buying a 3090 vs a mid-price companion subscription (~$180/yr) is ~27 months.

---

### Local fine-tuned models vs cloud options — detailed comparison

**vs companion apps (OurDream.ai and similar)**
Companion apps like OurDream are primarily visual platforms — image and video generation
from your RP character, voice options, and polished UX are their core differentiators.
For text RP, the underlying model is undisclosed and varies by tier. Base tier includes
platform auto-summarisation and optional pinned memories, so in-session coherence is a
combined effect of model + infrastructure — not raw context window. The ST equivalent is
ST with memory augmentation (vector storage, summary injection, lorebook) configured,
not bare ST. That said, the setup is manual in ST whereas OurDream handles it automatically.

The real advantage of local models is not coherence — companion app base-tier coherence,
boosted by auto-summary, is competitive with a local 12B on most sessions. The advantage
is **genre specificity**: your 17k-story corpus gives your fine-tuned model vocabulary,
power-dynamic logic, and tag adherence that no companion app has. From message one, your
model understands the genre; the companion app generates from general RP training data and
has no concept of your specific tag vocabulary or dynamic conventions. The subscription
makes sense if multimedia or zero-setup memory matters; for pure text RP with genre
precision, local models win clearly once trained, and 32B wins on prose depth.

**vs NovelAI Clio (~8B)**
Your fine-tuned 12B is clearly better. Clio is a well-tuned fiction model but 12B
outweighs it on prose variety and coherence, and your fine-tune has 17,000 stories
of specific corpus knowledge that Clio doesn't. The only advantage Clio has is
convenience — no training required.

**vs NovelAI Kayra (~13B)**
This is the closest comparison. Both are ~13B models trained on fiction corpora.
Kayra has better base prose training than Mistral Nemo (NovelAI has invested
heavily in their fine-tuning data quality), so raw prose output is roughly equal.
The meaningful difference is knowledge: your 12B knows your specific tag
vocabulary, genre conventions, and can use story bibles directly. Kayra knows
nothing about `hm + ds + in` as a tag set or what `stepfordized` means
narratively. For open-ended generation with a detailed prompt, they're comparable.
For structured generation guided by your archive tags and bibles, your 12B wins
clearly. Kayra costs $15/mo ongoing; your 12B is free after the one-time training.

**vs Venice.ai (70B)**
Venice wins on prose quality. A 70B model produces noticeably better sentence
variety, emotional range, and long-range coherence than any 12B — the parameter
gap is large enough that it shows consistently. However Venice has no genre
fine-tuning at all. Without a detailed system prompt it will write generic fiction,
handle genre vocabulary clumsily, and ignore the power-dynamic logic that
experienced readers expect. With a good system prompt (see the combination
workflow below), Venice produces excellent prose; your 12B produces correct-genre
output. The strongest results use both together.

**The key insight: genre knowledge vs prose quality are separate axes**

| | Genre knowledge | Prose quality |
|---|---|---|
| OurDream.ai | ★★☆☆☆ | ★★★☆☆ |
| Fine-tuned 8B (local) | ★★★★★ | ★★★☆☆ |
| Fine-tuned 12B (local) | ★★★★★ | ★★★½☆ |
| Fine-tuned 32B (local) | ★★★★★ | ★★★★☆ |
| NovelAI Kayra | ★★★★☆ | ★★★★☆ |
| Venice 70B | ★★☆☆☆ (unprompted) / ★★★★☆ (prompted) | ★★★★☆ |

Your fine-tuned models are the only options with both genre knowledge and decent-to-excellent
prose simultaneously, without extra prompting overhead. Cloud options either have broad fiction
training (NovelAI) or large-model prose quality (Venice) but require work to get both at once —
and neither knows your corpus.

---

### Cloud options in more detail

**NovelAI** is the most direct comparison to your fine-tune. They train their own
models specifically on fiction (including adult content) and have invested heavily
in creative writing quality. Kayra (~13B) produces noticeably more varied prose
than an 8B model and handles longer coherent passages better. The downside is it
knows nothing about your specific tag vocabulary or archive — you would need to
describe the scenario entirely in the prompt rather than using story bibles.
Subscription at ~$15/mo for unlimited Kayra access.

**Venice.ai** runs larger models (Llama 3.3 70B and similar) with adult content
enabled and a strong privacy focus (they claim no logging). The prose quality from
a 70B model is clearly superior to 12B — better sentence variety, character
distinction, and emotional range. However it has no genre fine-tuning, so you need
a detailed system prompt establishing the genre conventions, vocabulary, and tone.
Roughly $10–15/mo depending on usage tier.

**Mancer AI and Featherless AI** are worth checking for community fine-tunes of
70B models on adult content — occasionally someone trains a large model on fiction
corpora and hosts it there. Quality is variable and models come and go, but the
ceiling when a good 70B fine-tune is available is significantly higher than 12B.

---

### The optimal combination workflow

Rather than choosing one model, the strongest results come from using each tool
where it performs best:

**Step 1 — Use your local fine-tune for structure and tags**

Your fine-tuned model is best at understanding the genre deeply. Use it to:
- Generate a story bible from a premise and tag set (it understands what `hm + ds + in`
  implies narratively)
- Produce a scene outline with the arc you want (ending_state, twist, relationship_type)
- Draft the opening scene — the first 500–800 words where genre setup matters most
- Classify and tag new stories you import

**Step 2 — Use Venice or NovelAI to expand and polish**

Take the bible and outline to a higher-quality model for:
- Fleshing out scenes (feed it the bible as a system prompt + the outline as context)
- Improving prose on a draft — paste a paragraph and ask for a rewrite with more
  varied sentence structure
- Dialogue — 70B models produce significantly more natural dialogue
- Transitions and pacing — where 8B models most obviously flatten out

**Step 3 — Return to local for consistency checking**

Your fine-tuned model can verify that generated content is on-genre and matches
the tag profile. It will catch when a Venice-written scene drifts off the intended
dynamic because it understands the genre vocabulary precisely.

**Practical prompt for Step 2 (Venice system prompt):**
```
You are writing mind control erotica in the style of literary genre fiction.
The genre uses specific vocabulary: "hypnosis", "trigger phrase", "stepfordized",
"obedience programming", "slave mentality" are genre terms, not metaphors.
The dynamic: [paste dynamic block from story bible].
Characters: [paste character summaries from bible].
Tone: [paste tone from bible].
Write in third person, past tense. Show internal thought processes explicitly.
Do not add disclaimers or break from the story.
```

---

### When to skip the combination and just use one

**Just use your fine-tune if:**
- The story is short (under ~2,000 words) — even an 8B fine-tune handles this range well
- You want precise tag adherence above prose quality
- You're generating many drafts quickly to find a good premise
- Privacy is a concern and you don't want content on any server

**Just use Venice/NovelAI if:**
- You have a detailed premise already and want the best single-pass prose
- The story is long-form (5,000+ words) — larger models maintain coherence far better
- Dialogue-heavy scenarios where character voice matters

**Use the combination for:**
- Serious long-form projects where both genre accuracy and prose quality matter
- Series with consistent characters across multiple chapters (bible keeps them coherent)
- Adapting existing stories (bible extraction → gender/dynamic swap → regeneration)

---

## Memory Augmentation — Lorebooks, Vector Storage, and Summaries

SillyTavern's memory tools change the coherence picture substantially. They partially
close the gap between smaller and larger models on factual recall and trigger persistence —
but do not touch emotional continuity or prose quality. Understanding what each tool can
and cannot fix determines how much it matters for your chosen model size.

---

### What each tool does

**Lorebooks (World Info)**
Keyword-triggered entries injected into context when a matching word appears in the
current message or recent history. You define entries like `keys: pocket watch, spiral`
→ `[entry text]`. When those keywords appear, the entry is automatically prepended
near the top of the prompt, every single turn.

Best for: hard-coding established triggers, fixed character facts, relationship rules,
setting details. Anything that must never be forgotten regardless of context length.

**Vector storage**
Semantic search over your full conversation history. On each turn SillyTavern finds
the most relevant past messages and injects them into context. Available built-in in
recent ST versions (uses transformers.js locally, no extra server needed), or via
the Extras API + ChromaDB for older setups.

Best for: recalling specific past exchanges ("remember when you said..."), retrieving
facts established long ago that are not in the lorebook, bridging very long sessions.

**Periodic summaries (Summarize extension)**
Every N messages, a separate LLM call compresses recent conversation into a short
summary that sits permanently in context. The compressed summary replaces the raw
messages it covers.

Best for: sessions that would otherwise overflow the context window. A summary preserves
arc state ("she has now yielded to the spiral three times and is more susceptible")
even after the raw messages scroll out.

**Author's Note**
A short always-on injection inserted at a fixed position from the bottom of context
(default: 4 messages up, which gives it high weight). Not keyword-triggered — always
present every turn.

Best for: format reminders ("respond in first person, 2–4 sentences"), reinforcing the
subject's current role, anchoring the scene tone mid-session.

---

### Effect on session coherence by model size

Augmentation's impact on *factual recall* is large. Its impact on *emotional continuity*
is modest — you can tell a model the subject is at stage 2 of induction via a lorebook
entry, but whether it expresses that with genuine nuance depends on model quality, not memory.

| | 8B fine-tuned | 12B fine-tuned | 32B fine-tuned | OurDream.ai |
|---|---|---|---|---|
| **Factual recall — no augmentation** | ~20 messages | ~40 messages | ~80 messages | ~50–80 (model + auto-summary‡) |
| **Factual recall — full augmentation** | ~50–60 messages | ~80–100 messages | Context window limit† | ~100+ (pinned memories enabled) |
| **Emotional continuity — no augmentation** | ~15 messages | ~35 messages | ~75 messages | ~40–60 (model + auto-summary‡) |
| **Emotional continuity — full augmentation** | ~20–25 messages | ~45–50 messages | ~90–100 messages | ~80–100 (pinned memories enabled) |
| **Prose quality, augmented** | ★★★☆☆ (unchanged) | ★★★½☆ (unchanged) | ★★★★☆ (unchanged) | ★★★☆☆ (unchanged) |
| **Lorebook support** | ✅ | ✅ | ✅ | ❌ (pinned memories are a partial substitute) |
| **Vector storage** | ✅ | ✅ | ✅ | ❌ (auto-summary is a partial substitute) |
| **User control over memory** | ✅ full | ✅ full | ✅ full | ❌ opaque |
| **"Improved memory" available** | ✅ (built-in) | ✅ (built-in) | ✅ (built-in) | Coin paywall |

†At 32B with ctx=8,192, ~150+ active messages fit in the raw context window anyway.
Vector storage and summaries rarely needed except for very long multi-hour sessions.

‡OurDream base tier includes platform auto-summarisation and optional pinned memories.
"No augmentation" figures here reflect the system as it actually runs, not a hypothetical
bare-context configuration. The "full augmentation" row reflects pinned memories also enabled.

**Key finding:** augmentation closes most of the factual recall gap between model sizes.
It does not close the emotional continuity or prose quality gaps. A well-augmented 8B
approaches a bare 12B on factual recall and trigger persistence — but still reads like
an 8B in terms of nuance and response naturalness.

---

### Trigger persistence specifically

Hypnotic triggers are the most important thing to preserve across a long session —
and the place where lorebooks provide the clearest, most measurable benefit.

**Without a lorebook**, a trigger lives only in context history. Once it scrolls out
of the active context window (or the model simply stops attending to it), the trigger
is effectively forgotten. For an 8B this can happen as early as message 15.

**With a lorebook entry**, the trigger definition is injected on every relevant turn,
regardless of session length. Example:

```
Keys: spiral watch, the watch, silver spiral
Entry: [Established session trigger] When the subject sees or hears mention of the
spiral watch, she enters a light trance: eyes track the motion, breathing slows,
resistance softens. This was established at the start of the session. The effect
deepens each time it is used — she is currently at stage 2 of 4.
```

The lorebook ensures the model always knows the trigger exists. Model size then
determines how well it executes that knowledge:

| | 8B | 12B | 32B |
|---|---|---|---|
| **Trigger recall (with lorebook)** | ✅ reliable | ✅ reliable | ✅ reliable |
| **Trigger recall (without lorebook)** | Fails ~msg 15 | Fails ~msg 35 | Fails ~msg 75 |
| **Executes trigger correctly** | Usually | Reliably | Reliably, with nuance |
| **Tracks trigger deepening across session** | ❌ often resets | Partial | ✅ |
| **Subtly foreshadows / builds toward trigger** | ❌ | Partial | ✅ |
| **Varies triggered response naturally** | Formulaic | Moderate variety | High variety |

"Trigger deepening" — where repeated exposure produces progressively stronger responses —
is the hardest dynamic to preserve. You can update the lorebook entry mid-session
("now at stage 3 of 4") to manually track progression, which helps all model sizes.
A 32B integrates that state into subsequent responses more naturally than an 8B,
which may read the entry but respond formulaically regardless.

---

### Effect on cloud and commercial options

| | Companion app† (e.g. OurDream) | NovelAI (Kayra) | Venice.ai via API + ST |
|---|---|---|---|
| **Lorebook support** | ❌ | ✅ (native, core feature) | ✅ (via ST connection) |
| **Trigger hard-coding** | ❌ | ✅ | ✅ |
| **Vector storage** | ❌ | ❌ native (some interfaces add it) | ✅ (via ST) |
| **Author's Note** | ❌ | ✅ native | ✅ (via ST) |
| **Periodic summaries** | Partial (opaque, app-managed) | Via third-party tools | ✅ (via ST) |
| **User control over memory** | ❌ | ✅ | ✅ |
| **Cross-session memory** | ✅ (app-managed) | Partial | ❌ without extra tooling |

Companion apps' app-managed memory is good for retaining broad character facts across
many separate sessions — entity extraction and summarisation happen automatically. But you
cannot hard-code specific triggers, inject mid-session scene notes, retrieve a specific
past exchange, or control what gets remembered vs forgotten. The underlying model is
undisclosed and varies by tier; cheaper tiers likely use smaller models. The lorebook
trigger entry above is simply not possible in a companion app — the closest equivalent
is hoping the app's auto-extractor decides that trigger is worth storing.

NovelAI with Kayra is actually comparable to the SillyTavern experience in memory
tooling — native lorebook, Author's Note, and context injection. The gap is that Kayra
cannot be fine-tuned on your corpus, so genre vocabulary and tag adherence are weaker
regardless of how good your lorebook is. A well-crafted lorebook can inject genre
context explicitly ("this is MC erotica; the following terms are literal...") but it
is a workaround rather than baked-in knowledge.

Venice.ai used via their web interface has minimal memory tooling. Used via API with
a SillyTavern connection, you get the full ST stack — lorebooks, vector storage,
summaries — applied to a 70B model. That combination is the strongest available for
prose quality + memory augmentation, at the cost of no genre fine-tuning and no privacy.

---

### What augmentation can and cannot fix

**Augmentation fixes:**

| Problem | Fix |
|---|---|
| Character forgets established trigger | Lorebook hard-codes it |
| Model forgets character's name / relationship | Lorebook entry or character card |
| Model forgets what happened 40 messages ago | Vector storage retrieves it |
| Session runs over context window | Periodic summary compresses old turns |
| Model slips out of first-person or correct length | Author's Note anchors it |
| Established rule forgotten mid-session | Lorebook or manual Summary update |

**Augmentation does not fix:**

| Problem | Why augmentation cannot help |
|---|---|
| Prose variety flattens | Parameter ceiling — no memory tool increases sentence variety |
| Emotional depth is shallow | Model cannot respond with more nuance than its size allows |
| Model misses subtle implication | Reactivity is a model quality, not a memory problem |
| Triggered response feels formulaic | Model quality determines execution, not recall |
| Arc pacing rushes to resolution | Structural tendency of smaller models, not a memory failure |
| Character voice homogenises | Parameter ceiling — more context does not create more vocal distinctiveness |

The honest summary: augmentation lifts the floor significantly. It makes an 8B
meaningfully more consistent and brings it closer to a bare 12B on factual tasks.
It does not raise the ceiling — the ceiling is the model's parameters.

---

### Recommended setup by tier

**8B fine-tuned — hardware-constrained fallback only:**

Not recommended for RP with progressive mechanics. If you are running 8B because 12B
consistently OOMs on your hardware, apply all augmentation tools — the short coherence
window means every tool has a visible effect, though the quality ceiling remains.

- **Lorebook:** essential. Hard-code all established triggers with keyword sets, key
  character facts, established dynamic rules. Keep entries under 200 words. 5–10 entries
  covers most sessions. Update manually as triggers deepen ("now at stage X").
- **Author's Note:** always on, 4 messages from bottom. Short: role, format, current
  tone. Example: `[You are [name], the subject. Respond in first person. 2–3 sentences.
  Show internal state. Current stage: entering light trance.]`
- **Vector storage:** set up and use it. Retrieve 5–8 messages per turn. Most impactful
  at this tier because the raw context window is shortest.
- **Summaries:** use for sessions over ~30 messages. The 8B will generate summaries of
  moderate quality — keep them short (under 200 words) and manually correct if needed.

**12B fine-tuned — augmentation is enhancement, not crutch:**

- **Lorebook:** still important for trigger persistence; less critical for general character
  feel, which the fine-tune sustains independently longer.
- **Author's Note:** useful for format reminders; the chatbot fine-tune handles role mostly.
- **Vector storage:** useful for long sessions (60+ messages); for typical 30–50 message
  sessions the 12B context covers most of it without retrieval.
- **Summaries:** only for sessions over ~60 messages. 12B generates good summaries.

**32B fine-tuned — augmentation for triggers and very long sessions only:**

- **Lorebook:** use for trigger hard-coding. The model holds character and dynamic naturally;
  lorebook is assurance rather than necessity. Worth setting up once.
- **Author's Note:** minimal — only if you're getting format drift (too long / wrong person).
- **Vector storage:** rarely needed. 32B at ctx=8,192 actively holds ~6,000–8,000 words of
  conversation — enough for most sessions without retrieval.
- **Summaries:** only for very long multi-hour sessions. 32B summaries are high quality and
  preserve emotional texture and arc state well.

---

### All coherence extension techniques — summary and hardware compatibility

Everything that meaningfully extends how long a session stays coherent, in rough order
of impact-per-effort:

| Technique | What it costs | 3070 Ti (8B/12B) | Mac M4 (8B/12B) | Vast.ai 3090 (32B) |
|---|---|---|---|---|
| **Lorebook** | Setup time | ✅ Essential | ✅ Essential | ✅ Worth doing |
| **Message pinning** | Zero | ✅ Essential | ✅ Essential | ✅ Worth doing |
| **Dynamic lorebook updates** | Manual mid-session | ✅ High impact | ✅ High impact | ✅ Medium impact |
| **Context window increase** | VRAM | ⚠️ Limited (see below) | ⚠️ Limited (see below) | ✅ High impact |
| **Author's Note** | Setup time | ✅ Always on | ✅ Always on | ✅ Always on |
| **Vector storage** | ST setup (~5 min) | ✅ High impact | ✅ High impact | ✅ Medium impact |
| **Repetition penalty tuning** | Zero | ✅ Medium impact | ✅ Medium impact | ✅ Medium impact |
| **Context shift (llama.cpp)** | Zero (automatic) | ✅ Automatic | ✅ Automatic | ✅ Automatic |
| **Periodic summaries** | LLM call overhead | ✅ Medium (quality limited) | ✅ Medium | ✅ High quality |
| **Manual context editing** | Session effort | ✅ High impact | ✅ High impact | ✅ Medium impact |
| **Long-context native model** | VRAM | ❌ No 32B | ❌ No 32B | ✅ Medium impact |

**Context window limits by hardware:**

The single biggest lever after lorebooks is simply how much active context the model
can hold. This is a VRAM constraint, not a software one.

| Hardware | Model | Safe max ctx | Comfortable ctx | Notes |
|---|---|---|---|---|
| RTX 3070 Ti (8 GB) | 8B Q5_K_M | 12,288 | 8,192 | Guide default; 16,384 risks OOM |
| RTX 3070 Ti (8 GB) | 12B Q4_K_S | 8,192 | 6,144 | Guide default; already at ceiling |
| Mac M4 (16 GB unified) | 8B Q4_K_M | 16,384 | 8,192 | Higher context causes memory pressure |
| Mac M4 (16 GB unified) | 12B Q4_K_M | 8,192 | 6,144 | Shared with OS — headroom is tight |
| Vast.ai 3090 (24 GB) | 32B Q4_K_M | 12,288 | 8,192 | ~22 GB total at 12K; some hosts hit swap |
| Vast.ai 4090 (24 GB) | 32B Q4_K_M | 16,384 | 12,288 | Same VRAM as 3090 but faster |

Setting ctx in LM Studio / KoboldCPP: this is the "Context length" slider. Increasing
it is free if VRAM allows; increasing past the safe limit causes OOM mid-session.

**Message pinning** — SillyTavern lets you right-click any message and pin it. Pinned
messages are excluded from the context sliding window and stay in context permanently.
Use this on: the message where a trigger is first established, any message that defines
a key scene rule, and the opening scene-setter. Zero cost, high impact.

**Dynamic lorebook updates** — open the lorebook mid-session and edit the entry to
reflect current arc state, e.g. changing "trigger: light trance, stage 1" to "stage 3,
eyes glazed, offers minimal resistance". The model reads the updated entry on every
subsequent turn. More precise than vector storage for tracking gradual state changes.

**Repetition penalty (1.05–1.15)** — prevents the model looping on the same phrase
or idea, which is a local coherence failure rather than a memory failure but looks
similar. Already set to 1.05–1.1 in the guide's ST settings. Worth checking it is
active; a missing rep penalty is an easy source of apparent coherence loss.

**Context shift** — when the context window fills, llama.cpp (used by LM Studio and
KoboldCPP) discards roughly the oldest half and keeps recent history intact. This is
already happening automatically — messages don't hard-cut at the boundary, they fade
from oldest to newest. You cannot disable it, but you can influence which messages
survive longest by keeping sessions focused (verbose tangents fill the window faster).

**Long-context native models** — Qwen2.5-32B-Instruct supports up to 128K context
natively. In practice KV cache VRAM limits you to 12–16K on a 3090, but that is
already double the 8192 guide default. The model attends better to long-range content
within that window vs a model that was only trained on 4K context.

**Priority recommendation:**

On **3070 Ti** (12B recommended, Q4_K_S at ctx 6144): lorebook + message pinning + vector storage covers most
of your gap. Context window is already at ceiling; do not push it. Dynamic lorebook
updates give the most additional benefit for session cost.

On **Vast.ai 3090** (32B): increase ctx to 12,288 first (biggest single gain),
add lorebook for trigger hard-coding, pin key messages. Vector storage and summaries
are rarely needed unless sessions run over 2 hours.

---

## Pipeline B — Chatbot / SillyTavern RP Fine-Tune

This pipeline teaches a model to *be* the submissive subject in conversation — you
(the player) are the dominant, and the AI reacts to you. The result runs in
SillyTavern and responds directly, in first person, at conversational length.

### Role setup

| Role | Who | What they do |
|---|---|---|
| **user** | You (the player) | Dominant — gives commands, inductions, suggestions |
| **assistant** | The AI model | Subject — reacts, resists, yields, narrates internal state |

### What changes vs Pipeline A

The story fine-tune teaches the model to write 2,000-word narratives in third person.
That's the wrong skill for SillyTavern where you want short, reactive, first-person
responses from the subject. The chatbot fine-tune corrects this by training on
extracted dialogue turns with the correct dominant/submissive role split.

### B1 — Extract dialogue data (on Mac)

```bash
# Preview — see how many conversations would be extracted
python extract_chatbot_data.py --stats

# Full extraction — writes output/chat_data.jsonl
python extract_chatbot_data.py

# Optional: limit stories processed (faster, smaller dataset)
python extract_chatbot_data.py --limit 3000
```

**What the extractor does:**

Two extraction strategies run on each story:

1. **Scene pairs** — sliding window of 3 paragraphs becomes the "user" turn
   (dominant context/action), the next 2 paragraphs become the "assistant" response
   (subject reaction). Only included when the assistant block has submissive signals
   or dialogue.

2. **Dialogue runs** — finds consecutive paragraphs with quoted speech and frames
   them as alternating user/assistant turns:
   - Dominant-signalling paragraphs ("hypnot", "obey", "you will", "deeper") → **user**
   - Submissive-signalling paragraphs ("couldn't resist", "felt herself", "my mind") → **assistant**
   - Unclassified paragraphs → alternating, starting with user

**Expected output (full run, 18k stories):**
```
Stories processed:       18,252
Skipped (low dialogue):  1,057
Conversations extracted: 161,574
Avg turns per conv:      8.0
Total turns:             1,286,924
File size:               ~359 MB
```

Low-dialogue stories (fewer than 6 quoted exchanges) are skipped — they don't
provide enough signal for conversational training. Only ~5.8% of stories are
skipped, so coverage is very high.

**Optional: smaller/faster dataset**
```bash
# ~25k conversations — good quality, trains in ~3–4 hours on 3070 Ti
python extract_chatbot_data.py --limit 3000
```

### B2 — Validate on Windows (free, ~20 min)

Run a tiny smoke-test before committing to the full training run:

```cmd
cd C:\MCTraining
venv\Scripts\activate

# Copy chat_data.jsonl from Mac (USB or network share)
# Then run 50-step test:
python train_chat.py --train --epochs 1 --preset 3070
```

Watch for:
- "Loading model" — downloads Sunfall (~5GB, one time only)
- "Trainable: X.XM / 8000M" — LoRA applied
- "Step X/X, Loss: X.XX" — training running
- Loss starting above 2.0 and dropping

If OOM: the 3070 Ti preset uses 2048 seq_len which should fit. If it still OOMs,
add `--lora-r 4` to reduce LoRA memory.

### B3 — Full training on Windows RTX 3070 Ti

```cmd
python train_chat.py --train --preset 3070
```

No A100 needed — the 2048 token sequences fit comfortably in 8GB VRAM.

**Expected timeline:**
| Dataset size | Steps | Est. time |
|---|---|---|
| 25,000 conversations | ~9,000 | ~2.5 hours |
| 50,000 conversations | ~18,000 | ~5 hours |
| 161,574 conversations (full) | ~60,000 | ~16–20 hours |

Run the full dataset overnight, or cap at ~50k for a 5-hour run that's still excellent quality:
```cmd
python extract_chatbot_data.py --limit 5000
python train_chat.py --train --preset 3070
```

**Loss targets:**
- Start: ~2.0–2.3
- Good finish: ~0.8–1.1
- Stop early if loss plateaus or rises — 3 epochs is usually enough

Resume safely after interruption:
```cmd
python train_chat.py --train --resume
```

**Optional: faster run on A100 (~45 min, ~$1–2)**
```cmd
python train_chat.py --train --preset a100
```
The a100 preset uses 4096 token context so full multi-turn conversations fit
without truncation — slightly better quality for long RP sessions.

### B4 — Test before merging

```cmd
python train_chat.py --test
```

The test prompts three scenarios:
1. Resistant subject: "I can't be hypnotized"
2. Scene continuation: character entering the room nervously
3. Push-back: "I want you to stop"

Good output looks like:
- First person from the subject's perspective ("I tried to look away...", "Something was pulling at my thoughts...")
- Conversational length (2–5 sentences, not 800 words)
- Shows internal conflict, physical sensation, yielding
- Reacts directly to what the user said — not generic
- No "As an AI..." disclaimers, no third-person narration

If responses are still story-like (third person, long prose blocks), the dataset
may be too heavy on narrative paragraphs vs subject dialogue. Re-extract with
stricter settings:
```bash
python extract_chatbot_data.py --min-pairs 4
```

### B5 — Merge and convert to GGUF

```cmd
# Merge LoRA into full model (~24GB for 12B, ~15GB for 8B — takes ~5–10 min)
python train_chat.py --merge

# Convert to GGUF (requires llama.cpp)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt
python convert_hf_to_gguf.py ../output/chat_merged --outtype q4_k_s \
    --outfile ../output/chat_model_Q4_K_S.gguf
```

Output: `output/chat_model_Q4_K_S.gguf` (~6.5GB for 12B Q4_K_S; ~4.5GB for 8B Q4_K_S)

> **Note:** Q4_K_S is required for 12B on the 3070 Ti (Q4_K_M overflows VRAM). If you
> plan to run exclusively on Mac (16GB unified memory), you can use `--outtype q4_k_m`
> instead for slightly better quality (~7.0GB).

### B6 — Load in LM Studio and connect SillyTavern

1. Copy `chat_model_Q4_K_S.gguf` to Windows (for 3070 Ti serving) or Mac (for local use)
2. Open LM Studio → My Models → Add model from files → select the GGUF
3. Local Server tab → select model → Start Server (port 1234)
4. LM Studio settings for this model:
   - Prompt Format: **Mistral Nemo** (Dolphin Nemo / Magnum 12B — check model card if different)
   - Context: **6144** (12B on 3070 Ti — use 8192 for Mac-only use)
   - GPU layers: max
5. In SillyTavern → plug icon → Chat Completion → Custom (OpenAI-compatible)
   → URL: `http://localhost:1234/v1` (or your 3070 Ti LAN IP — see LAN Hosting section) → Connect
6. Instruct Mode → **Mistral Nemo** template (or match to the model card)
7. Load your character card — the card's system prompt overrides the training
   system prompt, so your Evelyn/Sophie persona takes full effect

### B7 — SillyTavern settings for the chatbot LoRA

The chatbot LoRA was trained for shorter, reactive responses from the subject's
perspective. Match your ST settings to what the model expects:

```
Temperature:        0.88 – 0.95
Min-P:              0.05            (disable Top-P — set to 1.0)
Repetition Penalty: 1.05 – 1.1
Max New Tokens:     200 – 400       (subject reactions, not story paragraphs)
Context Size:       8192
```

**Character card setup:**
Your character card describes the *subject* — the person being hypnotized/controlled.
You (the player) are the dominant. The AI plays the subject reacting to you.
The card's system prompt overrides the training system prompt, so write it in
second person or first person from the subject's perspective:
```
[You are Kara, a psychology student who agreed to a hypnosis demonstration.
You are curious but skeptical, with a strong will that makes you resistant to
suggestion — at first. You speak your thoughts aloud as they happen, describing
what you feel and sense as the induction progresses.]
```

**Author's Note** — inject this 4 messages from the bottom:
```
[You are the subject. Respond in first person. 2–4 sentences.
Show what you feel and sense. React to what was just said or done.
Do not narrate in third person.]
```

### Stacking: chatbot LoRA on top of story LoRA

If you've already trained Pipeline A (story fine-tune), you can apply the chatbot
LoRA as a second pass on top of the merged story model rather than on the base
Sunfall model. This gives you a model that both understands the genre deeply AND
responds conversationally.

```cmd
# Point train_chat.py at your story-fine-tuned merged model
python train_chat.py --train --model ./output/merged_model
```

The two LoRAs are combined in the merge step. The result is the strongest
possible base for SillyTavern RP: genre vocabulary from 17k stories + conversational
behaviour from the dialogue extraction.

---

### Chatbot pipeline troubleshooting

**Responses still in third person / too long**
- The story prose is overwhelming the dialogue signal. Re-extract with stricter filters:
  ```bash
  python extract_chatbot_data.py --min-pairs 4
  ```
- Or reduce `SCENE_WINDOW` from 3 to 2 in `extract_chatbot_data.py` to shorten
  the "user" context blocks.

**Loss doesn't drop below 1.5**
- Lower learning rate: `--learning-rate 5e-5`
- Or increase epochs: `--epochs 5`

**Loss drops below 0.5 early (overfitting)**
- Dataset too small. Lower `--min-pairs` to include more conversations, or
  reduce epochs to 2.

**Model loses its Sunfall prose quality**
- Reduce LoRA rank: `--lora-r 4` (less aggressive adaptation)
- Or fewer epochs: `--epochs 2`

**OOM on 3070 Ti**
- Already at seq_len=2048 which is the minimum useful. Try: `--lora-r 4`

---

## Production Run Checklist

One-screen reference for A100 rentals. Print or keep open in a second window.

### Before renting

- [ ] Dataset report reviewed — no unexpected exclusions or size changes
- [ ] Validation split frozen and not included in training data
- [ ] Training JSONL hash saved: `shasum -a 256 output/train_data.jsonl`
- [ ] Base model revision pinned (note the commit hash from Hugging Face)
- [ ] Prompt template verified against model card
- [ ] Local validation (Phase 1) passed within last week
- [ ] Expected cost calculated: hourly rate × (upload + setup + train + merge + download)
- [ ] Download destination on local machine confirmed and has sufficient space

### On the instance

- [ ] Exact GPU model confirmed (`nvidia-smi`)
- [ ] VRAM confirmed — matches expected
- [ ] Disk space confirmed (`df -h`)
- [ ] Environment saved: `pip freeze > requirements-lock.txt`
- [ ] `tmux` session started before beginning any long command
- [ ] One-batch pilot run completed — loss is finite and decreasing
- [ ] Peak VRAM recorded from `nvidia-smi` during pilot
- [ ] Checkpoint written and readable
- [ ] Resume path confirmed (test interrupting and resuming if feasible)

### Before destroying the instance

- [ ] LoRA adapter folder downloaded and verified (check expected ~250–400MB)
- [ ] Training logs downloaded
- [ ] Config / preset file downloaded
- [ ] Hash of adapter files recorded
- [ ] Adapter loads locally without errors (`python train.py --test` or equivalent)
- [ ] No needed data remains only on the instance
- [ ] Instance **destroyed** (not just stopped) to halt billing

---

## Change Log

### Version 2.0 — 2026-07-13

- Changed recommended base from 8B to 12B as minimum viable for RP
- Added document metadata header and evidence labels
- Added Current Recommended Path summary table
- Added Known Facts and Open Questions section
- Added Authoritative Model Matrix
- Added Adapter Stacking Options section
- Added Prompt Template Reference table
- Added Canonical Quantization and Context table
- Added Volatile Pricing Snapshot section
- Added LAN security note
- Added go/no-go checklist before A100 rental
- Added Production Run Checklist
- Updated Pipeline B description to reflect 12B training requirements
- Updated optimal 3070 Ti settings: 12B Q4_K_S listed first, 8B demoted to fallback
- Updated recommended path: 12B chat LoRA as step 1, 8B demoted to step 5
- Updated LAN performance table: 12B row first
- Updated KoboldCPP GPU layers for 12B (40, not 33)
- Updated GGUF conversion to Q4_K_S as default for 12B
- Updated B6 LM Studio settings for Mistral Nemo template and ctx 6144
- Clarified Mac speed contradiction: 17 tok/s (GGUF/measured) vs 50–60 tok/s (MLX/estimated)
- Clarified arc tagging uses 8B even when 12B is the RP model
- Elevated data-collator workaround severity warning
