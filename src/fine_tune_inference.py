import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class FineTuneInference:
    def __init__(self,
                 base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
                 adapter_dir="sample-size-sft-lora",
                 quantized: bool | None = None,  # None = auto (best effort)
                 attn_impl: str | None = None,  # "sdpa" or "flash_attention_2"
                 compile_model: bool = True,
                 ):
        self.has_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.has_cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Decide quantization
        if quantized is None:
            quantized = False
        # Choose attention implementation
        if attn_impl is None and self.has_cuda:
            attn_impl = "sdpa"
        self.attn_impl = attn_impl
        self.quantized = bool(quantized and self.has_cuda)
        self.base_model = self._load_base(base_model_name)
        self.model = PeftModel.from_pretrained(self.base_model, adapter_dir)
        self.model.eval()
        self.model.config.use_cache = True
        self.model.generation_config.use_cache = True
        if self.has_cuda and compile_model:
            # Compile can help for repeated inference; first call slower
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
        self.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        # self.base_model = AutoModelForCausalLM.from_pretrained(
        #     base_model_name,
        #     dtype=torch.bfloat16 if has_cuda else torch.float32,
        #     device_map=None,
        #     low_cpu_mem_usage=True,
        # ).to(device)
        # self.tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        # if self.tok.pad_token is None:
        #     self.tok.pad_token = self.tok.eos_token
        # self.model = PeftModel.from_pretrained(self.base_model, adapter_dir)
        # self.model.eval()
        # self.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _load_base(self, base_model_name: str):
        if self.quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            return AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="cuda",  # IMPORTANT for bnb
                dtype=torch.bfloat16,
                attn_implementation=self.attn_impl,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.bfloat16 if self.has_cuda else torch.float32,
                attn_implementation=self.attn_impl if self.has_cuda else None,
                low_cpu_mem_usage=True,
            )
            return model.to(self.device)

    # def predict(self, input_messages: list, max_tokens: int = 8):
    #     inp = self.tok.apply_chat_template(
    #         input_messages,
    #         add_generation_prompt=True,
    #         return_tensors="pt").to(self.model.device)
    #     with torch.no_grad():
    #         out = self.model.generate(
    #             **inp,
    #             max_new_tokens=max_tokens,
    #             do_sample=False
    #         )
    #
    #     gen = out[0, inp["input_ids"].shape[1]:]
    #     text = self.tok.decode(gen, skip_special_tokens=True).strip()
    #
    #     return text

    def predict(self, input_messages, max_tokens: int = 8):
        return self.predict_batch([input_messages], max_tokens=max_tokens)[0]

    def predict_batch(self, batch_messages, max_tokens: int = 8):
        rendered = [
            self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            for msgs in batch_messages
        ]
        enc = self.tok(
            rendered,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=8,
        )

        main_device = next(self.model.parameters()).device
        enc = {k: v.to(main_device) for k, v in enc.items()}

        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
            )

        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        return [
            self.tok.decode(out[i, prompt_lens[i]:], skip_special_tokens=True).strip()
            for i in range(out.shape[0])
        ]

    # def predict_batch(self, batch_messages: list[list[dict]], max_tokens: int = 8) -> list[str]:
    #     # Render each conversation to text (fast), then tokenize as a batch
    #     rendered = [
    #         self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    #         for msgs in batch_messages
    #     ]
    #     enc = self.tok(
    #         rendered,
    #         return_tensors="pt",
    #         padding=True,
    #         # truncation=True,
    #     )
    #
    #     main_device = next(self.model.parameters()).device
    #     enc = {k: v.to(main_device) for k, v in enc.items()}
    #
    #     # Generate for the whole batch
    #     with torch.inference_mode():
    #         out = self.model.generate(
    #             **enc,
    #             max_new_tokens=max_tokens,
    #             do_sample=False,
    #             use_cache=True,
    #         )
    #     prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
    #
    #     results = []
    #     for i, prompt_len in enumerate(prompt_lens):
    #         gen_ids = out[i, prompt_len:]  # remove the prompt
    #         results.append(self.tok.decode(gen_ids, skip_special_tokens=True).strip())
    #
    #     return results

    # def evaluate_and_score(self, input_message_list: list[list[dict]], expected_results: list[str]):
    #     batch_preds = self.predict_batch(input_message_list)
    #     emb1 = self.st_model.encode(batch_preds, convert_to_tensor=True, normalize_embeddings=True)
    #     emb2 = self.st_model.encode(expected_results, convert_to_tensor=True, normalize_embeddings=True)
    #     sims = util.pairwise_cos_sim(emb1, emb2)
    #
    #     return sims.mean().item()

    def evaluate_and_score(self, input_message_list, expected_results):
        preds = self.predict_batch(input_message_list)
        emb1 = self.st_model.encode(preds, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
        emb2 = self.st_model.encode(expected_results, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.pairwise_cos_sim(emb1, emb2)
        return sims.mean().item()

    def score_predictions(self, evaluated_file_path: str):
        file_path = Path(evaluated_file_path)
        df = pd.read_csv(file_path).astype(str)
        if "predicted" in df.columns:
            df = df.rename(columns={"predicted": "predictions"})
        emb1 = self.st_model.encode(df['actual'].tolist(), convert_to_tensor=True, normalize_embeddings=True)
        emb2 = self.st_model.encode(df['predictions'].tolist(), convert_to_tensor=True, normalize_embeddings=True)
        sims = util.pairwise_cos_sim(emb1, emb2)
        df['similarity'] = sims.tolist()
        new_file_name = file_path.stem + '_modified.csv'
        df.to_csv(file_path.with_name(new_file_name), index=False)

        return sims.mean().item()
