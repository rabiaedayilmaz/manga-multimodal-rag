import argparse
from typing import Optional
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import time


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_first_turn_msgs(image: Image.Image, prompt: str):
    return [{'role': 'user', 'content': [image, prompt]}]


class MiniCPMV4Answerer:
    def __init__(self,
                 model_name: str = "openbmb/MiniCPM-V-4",
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            attn_impl = "sdpa"
        elif self.device == "cuda":
            dtype = torch.float16
            attn_impl = "sdpa"
        else:
            dtype = torch.float32
            attn_impl = "eager"

        print(f"Loading model on {self.device} with dtype={dtype}, attn={attn_impl}...")

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attn_impl
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )

    def answer(self,
               image_path: str,
               prompt: str,
               sampling: Optional[bool] = None,
               temperature: Optional[float] = None,
               top_p: Optional[float] = None,
               top_k: Optional[int] = None) -> str:

        image = load_image(image_path)
        msgs = build_first_turn_msgs(image, prompt)

        gen_kwargs = {}
        if sampling:
            gen_kwargs["sampling"] = True
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)

        print(f"Starting inference with prompt: {prompt}")
        start = time.time()

        with torch.inference_mode():
            out = self.model.chat(
                msgs=msgs,
                image=image,
                tokenizer=self.tokenizer,
                **gen_kwargs
            )

        print(f"Finished in {time.time() - start:.2f}s")
        return out if isinstance(out, str) else str(out)


def main():
    ap = argparse.ArgumentParser(description="MiniCPM-V-4 VLM inference")
    ap.add_argument("--image_path", required=True, type=str, help="Image file path")
    ap.add_argument("--prompt", type=str, default=(
        "Describe this manga panel in detail: characters, setting, and events. "
        "If possible, estimate which chapter or volume it belongs to, and justify your guess."
    ))
    ap.add_argument("--sampling", action="store_true", help="Enable sampling (optional)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    args = ap.parse_args()

    qa = MiniCPMV4Answerer()
    response = qa.answer(
        image_path=args.image_path,
        prompt=args.prompt,
        sampling=args.sampling,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )

    print("\nMiniCPM-V-4 Model Response\n")
    print(response)


if __name__ == "__main__":
    main()
