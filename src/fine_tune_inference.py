import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class FineTuneInference:
    def __init__(self, base_model_name="Qwen/Qwen2.5-1.5B-Instruct", adapter_dir="sample-size-sft-lora"):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=None,
            low_cpu_mem_usage=False,
        )
        self.tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = PeftModel.from_pretrained(self.base_model, adapter_dir)
        self.model.to(device)
        self.model.eval()
        # self.model.generation_config = GenerationConfig.from_model_config(self.model.config)
        # self.model.generation_config.do_sample = False
        # # Remove sampling knobs to get rid of warnings
        # self.model.generation_config.temperature = None
        # self.model.generation_config.top_p = None
        # self.model.generation_config.top_k = None

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
