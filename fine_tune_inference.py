import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class FineTuneInference:
    def __init__(self, base_model_name="Qwen/Qwen2.5-1.5B-Instruct", adapter_dir="sample-size-sft-lora"):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=None,
            low_cpu_mem_usage=False,
        )
        self.tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = PeftModel.from_pretrained(self.base_model, adapter_dir)
        self.model.eval()
        self.model.to(device)

    def predict(self, input_messages: list, max_tokens: int = 8):
        inp = self.tok.apply_chat_template(input_messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        # if hasattr(self.tok, "apply_chat_template"):
        #     # msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        #     inp = self.tok.apply_chat_template(input_messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        # else:
        #     inp = self.tok((system + "\n\n" + user), return_tensors="pt").to(self.model.device)

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


# def first_int(text: str):
#     m = re.search(r"-?\d+", text)
#     return int(m.group(0)) if m else None
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
#     ap.add_argument("--adapter_dir", default="sample-size-sft-lora")
#     ap.add_argument("--population_size", type=int, required=True)
#     ap.add_argument("--confidence_level", type=float, default=0.9)
#     ap.add_argument("--tolerable_error", type=float, default=0.1)
#     ap.add_argument("--assumed_p", type=float, default=0.5)
#     ap.add_argument("--used_fpc", action="store_true")
#     ap.add_argument("--rounding", default="ceil")
#     args = ap.parse_args()
#
#     tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token
#
#     base = AutoModelForCausalLM.from_pretrained(
#         args.base_model,
#         torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#         device_map="auto",
#     )
#     model = PeftModel.from_pretrained(base, args.adapter_dir)
#     model.eval()
#
#     system = "You are an expert data analyst. Return ONLY the final sample size as an integer."
#     user = (
#         f"Population size: {args.population_size}\n"
#         f"Confidence level: {args.confidence_level}\n"
#         f"Tolerable error rate: {args.tolerable_error}\n"
#         f"Assumed probability of success: {args.assumed_p}\n"
#         f"Rounding: {args.rounding}\n"
#         f"Used FPC: {bool(args.used_fpc)}\n\n"
#         "Return ONLY the final sample size.")
#
#     if hasattr(tok, "apply_chat_template"):
#         msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
#         inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
#     else:
#         inp = tok((system + "\n\n" + user), return_tensors="pt").to(model.device)
#
#     with torch.no_grad():
#         out = model.generate(
#             inp,
#             max_new_tokens=8,
#             do_sample=False,  # greedy for determinism
#             eos_token_id=tok.eos_token_id,
#             pad_token_id=tok.eos_token_id,
#         )
#
#     gen = out[0, inp.shape[-1]:]
#     text = tok.decode(gen, skip_special_tokens=True).strip()
#     print("raw:", text)
#     print("parsed_int:", first_int(text))
#
#
# if __name__ == "__main__":
#     main()
