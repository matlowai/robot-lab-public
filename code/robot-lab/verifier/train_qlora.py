"""4-bit QLoRA fine-tune of Gemma 4 12B (gemma4_unified) as a robot-skill verifier, with Unsloth.

Operator requirement (Dylan, 2026-10-06): "You will need 4 bit qlora unsloth training for a 12b model".
Two arms on the same data/split and the same QLoRA settings (operator follow-up, same day):
  --arm A  LoRA on the language model only (vision side frozen)              [default]
  --arm B  LoRA on language + vision layers (Unsloth finetune_vision_layers=True)
Gemma 4 12B "Unified" is encoder-free: image patches are projected straight into the LM (Gemma4UnifiedVisionEmbedder).
Unsloth's finetune_vision_layers=True selects nothing there, so arm B targets the embedder's two Linears explicitly
(see LANG_RE / VIS_RE below). The exact LoRA-wrapped module names per arm are written to lora_modules.json.

Prompt format = inference format: the user turn is rendered by the processor's chat template with
add_generation_prompt=True (Gemma 4 appends an EMPTY thought channel when thinking is off), then the JSON answer +
'<turn|>'. Loss is on the answer tokens only. Images are fed at their native 512x256 (no collator resize), the same
bytes the eval sends.

usage: train_qlora.py --arm A --data <dataset dir> --out <dir> [--base <hf id or path>] [--epochs 1] ...
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["A", "B"], required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="google/gemma-4-12B-it")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--max_steps", type=int, default=-1)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=16)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--ga", type=int, default=2)
ap.add_argument("--max_train", type=int, default=0, help="cap train samples (smoke tests)")
ap.add_argument("--seed", type=int, default=3407)
ap.add_argument("--exclude_object", default="", help="leave-one-object-out probe: drop this object from train/val")
args = ap.parse_args()

import unsloth  # noqa: E402,F401  (must precede transformers)
from unsloth import FastVisionModel  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
model, processor = FastVisionModel.from_pretrained(args.base, load_in_4bit=True, use_gradient_checkpointing="unsloth")
t_load = time.time() - t0
# Arm B target: Unsloth's finetune_vision_layers=True matched ZERO modules on gemma4_unified (box0 smoke test
# 2026-10-06: 328 LoRA modules, all language_model, identical to arm A) because the 12B Unified model has no vision
# tower. Its whole vision side is model.embed_vision (49.9M params): patch_dense Linear(6912->3840) and
# multimodal_embedder.embedding_projection Linear(3840->3840) (listed from the config on the meta device). Arm B
# therefore = arm A's exact language targets + LoRA on those two embedder Linears, same r/alpha/lr.
LANG_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
VIS_RE = r".*embed_vision\.(patch_dense|multimodal_embedder\.embedding_projection)"
peft_kw = dict(finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True, r=args.r,
               lora_alpha=args.alpha, lora_dropout=0.0, bias="none", random_state=args.seed, use_rslora=False)
if args.arm == "A":
    model = FastVisionModel.get_peft_model(model, finetune_vision_layers=False, **peft_kw)
else:
    model = FastVisionModel.get_peft_model(model, finetune_vision_layers=True, target_modules=f"({LANG_RE})|({VIS_RE})",
                                           **peft_kw)
lora_mods = sorted({n.rsplit(".lora_A", 1)[0] for n, _ in model.named_parameters() if ".lora_A" in n})
vision_mods = [m for m in lora_mods if any(k in m.lower() for k in ("vision", "visual", "embed_vision", "patch", "mm_"))]
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
json.dump(dict(arm=args.arm, n_lora_modules=len(lora_mods), n_vision_like=len(vision_mods), vision_like=vision_mods,
               sample=lora_mods[:20] + lora_mods[-20:], trainable_params=n_train),
          open(out / "lora_modules.json", "w"), indent=1)
print(f"ARM {args.arm}: {len(lora_mods)} LoRA modules, {len(vision_mods)} vision-like, trainable={n_train:,}; "
      f"load {t_load:.0f}s", flush=True)
if args.arm == "B" and len(vision_mods) != 2:
    raise SystemExit(f"arm B expected 2 embed_vision LoRA modules, got {vision_mods} -- refusing to train a fake arm B")
if args.arm == "A" and vision_mods:
    raise SystemExit(f"arm A must not touch vision modules, got {vision_mods}")

tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
END = "<turn|>"


def load_rows(p, cap=0):
    rows = [json.loads(line) for line in open(p) if line.strip()]
    return rows[:cap] if cap else rows


train_rows = load_rows(Path(args.data) / "train.jsonl", args.max_train)
val_rows = load_rows(Path(args.data) / "val.jsonl")
if args.exclude_object:
    train_rows = [r for r in train_rows if r["object"] != args.exclude_object]
    val_rows = [r for r in val_rows if r["object"] != args.exclude_object]
val_rows = val_rows[:256 if not args.max_train else 16]
print(f"train rows {len(train_rows)} val rows {len(val_rows)} exclude_object={args.exclude_object!r}", flush=True)


def prompt_text(prompt):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


class Collator:
    def __call__(self, rows):
        imgs = [[Image.open(r["image"]).convert("RGB")] for r in rows]
        prompts = [prompt_text(r["prompt"]) for r in rows]
        fulls = [p + r["answer"] + END for p, r in zip(prompts, rows)]
        tok.padding_side = "right"
        batch = processor(text=fulls, images=imgs, return_tensors="pt", padding=True)
        plen = [processor(text=[p], images=[im], return_tensors="pt")["input_ids"].shape[1] for p, im in zip(prompts, imgs)]
        labels = batch["input_ids"].clone()
        for i, n in enumerate(plen):
            labels[i, :n] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


# sanity: one rendered example + label span, written for audit
c = Collator()
b = c(train_rows[:1])
sup = b["labels"][0][b["labels"][0] != -100]
(out / "example_render.txt").write_text(
    "PROMPT+ANSWER (decoded, image tokens collapsed):\n" + tok.decode(b["input_ids"][0]).replace("<|image|>" * 4, "")[:3000]
    + "\n\nSUPERVISED TOKENS:\n" + tok.decode(sup) + f"\n\nseq_len={b['input_ids'].shape[1]} keys={list(b.keys())}\n")
print("SUPERVISED:", repr(tok.decode(sup)), "seq_len", b["input_ids"].shape[1], flush=True)

FastVisionModel.for_training(model)
steps_per_epoch = math.ceil(len(train_rows) / (args.bs * args.ga))
targs = TrainingArguments(
    output_dir=str(out / "ckpt"), per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.ga,
    num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr, warmup_steps=max(5, steps_per_epoch // 20),
    lr_scheduler_type="linear", optim="adamw_8bit", weight_decay=0.001, logging_steps=5, save_strategy="steps",
    save_steps=max(50, steps_per_epoch // 2), save_total_limit=2, bf16=True, seed=args.seed, report_to=[],
    remove_unused_columns=False, dataloader_num_workers=4, per_device_eval_batch_size=args.bs,
    eval_strategy="no", gradient_checkpointing=False,  # Unsloth already enabled its own checkpointing
)


class RowDS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


trainer = Trainer(model=model, args=targs, train_dataset=RowDS(train_rows), eval_dataset=RowDS(val_rows),
                  data_collator=Collator())
t1 = time.time()
res = trainer.train()
t_train = time.time() - t1
ev = trainer.evaluate()
model.save_pretrained(str(out / "adapter"))
processor.save_pretrained(str(out / "adapter"))
summary = dict(arm=args.arm, exclude_object=args.exclude_object, base=args.base, load_in_4bit=True, quant="bitsandbytes nf4 (Unsloth load_in_4bit)",
               r=args.r, alpha=args.alpha, lr=args.lr, bs=args.bs, ga=args.ga, epochs=args.epochs,
               max_steps=args.max_steps, train_samples=len(train_rows), val_samples=len(val_rows),
               global_steps=res.global_step, train_loss=res.training_loss, eval=ev, train_wall_s=round(t_train, 1),
               load_wall_s=round(t_load, 1), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               log_history=trainer.state.log_history)
import importlib.metadata as md  # noqa: E402
summary["versions"] = {p: md.version(p) for p in ["unsloth", "unsloth_zoo", "transformers", "torch", "bitsandbytes",
                                                   "peft", "trl", "accelerate"]}
json.dump(summary, open(out / "train_summary.json", "w"), indent=1)
print("TRAIN_DONE", json.dumps({k: v for k, v in summary.items() if k != "log_history"}), flush=True)
