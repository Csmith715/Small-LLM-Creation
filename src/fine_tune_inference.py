import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class FineTuneInference:
    def __init__(self, base_model_name="Qwen/Qwen2.5-1.5B-Instruct", adapter_dir="sample-size-sft-lora"):
        has_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if has_cuda else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16 if has_cuda else torch.float32,
            device_map=None,  # <- key change
            low_cpu_mem_usage=True,
        ).to(device)

        self.tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = PeftModel.from_pretrained(self.base_model, adapter_dir)
        self.model.eval()

        self.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # def __init__(self, base_model_name="Qwen/Qwen2.5-1.5B-Instruct", adapter_dir="sample-size-sft-lora"):
    #     # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #     has_cuda = torch.cuda.is_available()
    #     if has_cuda:
    #         torch.backends.cuda.matmul.allow_tf32 = True
    #     self.base_model = AutoModelForCausalLM.from_pretrained(
    #         base_model_name,
    #         dtype=torch.bfloat16 if has_cuda else torch.float32,
    #         device_map='auto',
    #         low_cpu_mem_usage=True
    #     )
    #     self.tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    #     if self.tok.pad_token is None:
    #         self.tok.pad_token = self.tok.eos_token
    #     self.model = PeftModel.from_pretrained(self.base_model, adapter_dir)
    #     # self.model.to(device)
    #     self.model.eval()
    #     self.st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def predict(self, input_messages: list, max_tokens: int = 8):
        inp = self.tok.apply_chat_template(
            input_messages,
            add_generation_prompt=True,
            return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_tokens,
                do_sample=False,
                # eos_token_id=self.tok.eos_token_id,
                # pad_token_id=self.tok.eos_token_id,
            )

        gen = out[0, inp["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True).strip()

        return text

    def predict_batch(self, batch_messages: list[list[dict]], max_tokens: int = 8) -> list[str]:
        # Render each conversation to text (fast), then tokenize as a batch
        rendered = [
            self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            for msgs in batch_messages
        ]
        enc = self.tok(
            rendered,
            return_tensors="pt",
            padding=True,
            # truncation=True,
        )

        main_device = next(self.model.parameters()).device
        enc = {k: v.to(main_device) for k, v in enc.items()}

        # Generate for the whole batch
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
            )
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

        results = []
        for i, prompt_len in enumerate(prompt_lens):
            gen_ids = out[i, prompt_len:]  # remove the prompt
            results.append(self.tok.decode(gen_ids, skip_special_tokens=True).strip())

        return results

    def evaluate_and_score(self, input_message_list: list[list[dict]], expected_results: list[str]):
        batch_preds = self.predict_batch(input_message_list)
        emb1 = self.st_model.encode(batch_preds, convert_to_tensor=True, normalize_embeddings=True)
        emb2 = self.st_model.encode(expected_results, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.pairwise_cos_sim(emb1, emb2)

        return sims.mean().item()

    def score_predictions(self, evaluated_file_path: str):
        df = pd.read_csv(evaluated_file_path).astype(str)
        if "predicted" in df.columns:
            df = df.rename(columns={"predicted": "predictions"})
        emb1 = self.st_model.encode(df['actual'].tolist(), convert_to_tensor=True, normalize_embeddings=True)
        emb2 = self.st_model.encode(df['predictions'].tolist(), convert_to_tensor=True, normalize_embeddings=True)
        sims = util.pairwise_cos_sim(emb1, emb2)

        return sims.mean().item()
