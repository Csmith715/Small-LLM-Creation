import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import BitsAndBytesConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train_jsonl", default="train.jsonl")
    ap.add_argument("--val_jsonl", default="val.jsonl")
    ap.add_argument("--out_dir", default="sample-size-sft-lora")
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset("json", data_files={"train": args.train_jsonl, "validation": args.val_jsonl})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    has_cuda = torch.cuda.is_available()
    # has_mps = torch.backends.mps.is_available()
    # device = "cuda" if has_cuda else "mps" if has_mps else "cpu"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if has_cuda else torch.float32,
    )
    attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map='auto',  # Need to use None and uncomment the device line above if run locally
        quantization_config=bnb_config,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    sft_config = SFTConfig(
        output_dir=args.out_dir,
        seed=args.seed,
        max_length=args.max_seq_len,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=10,
        bf16=has_cuda,
        fp16=False,
        packing=False,
        report_to="none",
        optim="paged_adamw_8bit",
        dataset_text_field="messages",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_config,
        args=sft_config,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
