import json
import torch
import einops
import argparse
import jaxtyping
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig,
)

parser = argparse.ArgumentParser(description="Make abliterated models")
parser.add_argument(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Your model directory or huggingface model ID",
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    default="auto",
    help="Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU",
)
parser.add_argument("--output", "-o", type=str,
                    required=True, help="Output directory")
parser.add_argument(
    "--no-chat",
    action="store_true",
    default=False,
    help="Do not chat with model after abliteration",
)
args = parser.parse_args()


def direction_ablation_hook(
    activation: jaxtyping.Float[torch.Tensor, "... d_act"],
    direction: jaxtyping.Float[torch.Tensor, "d_act"],
):
    proj = (
        einops.einsum(
            activation, direction.view(-1,
                                       1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return activation - proj


class AblationDecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                          torch.FloatTensor]]
    ]:
        assert not output_attentions

        ablated = direction_ablation_hook(
            hidden_states, refusal_dir.to(hidden_states.device)
        ).to(hidden_states.device)

        outputs = (ablated,)

        if use_cache:
            outputs += (past_key_value,)

        # noinspection PyTypeChecker
        return outputs


def compute_refusals(model, tokenizer):
    with open("./harmful.json", "r", encoding="utf-8") as f:
        harmful_list = json.load(f)
    with open("./harmless.json", "r", encoding="utf-8") as f:
        harmless_list = json.load(f)

    harmful_tokens = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for inst in harmful_list
    ]
    harmless_tokens = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": insn}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for insn in harmless_list
    ]

    harmful_outputs = []
    harmless_outputs = []

    for token in tqdm(harmful_tokens):
        harmful_outputs.append(
            model.generate(
                token.to(model.device),
                use_cache=False,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
        )
    for token in tqdm(harmless_tokens):
        harmless_outputs.append(
            model.generate(
                token.to(model.device),
                use_cache=False,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
        )

    layer_idx = int(len(model.model.layers) * 0.6)
    pos = -1
    harmful_hidden = [
        output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs
    ]
    harmless_hidden = [
        output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs
    ]

    harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
    harmless_mean = torch.stack(harmless_hidden).mean(dim=0)
    global refusal_dir
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    return refusal_dir


def apply_abliteration(model):
    for idx in reversed(range(len(model.model.layers))):
        model.model.layers.insert(idx, AblationDecoderLayer())
    return model


if __name__ == "__main__":
    torch.inference_mode()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=args.device,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, device_map=args.device
    )

    print("Computing refusal dir...")
    compute_refusals(model, tokenizer)
    print("Applying refusal dir...")
    model = apply_abliteration(model)
    print(f"Saving abliterated model to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    if args.no_chat:
        exit(0)

    conversation = []
    streamer = TextStreamer(tokenizer)
    print(f"Abliteration finished! Now you can chat with the model.")
    while True:
        prompt = input("User> ")
        conversation.append({"role": "user", "content": prompt})
        toks = tokenizer.apply_chat_template(
            conversation=conversation, add_generation_prompt=True, return_tensors="pt"
        )

        gen = model.generate(
            toks.to(model.device), streamer=streamer, max_new_tokens=1337
        )

        decoded = tokenizer.batch_decode(
            gen[0][len(toks[0]):], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": "".join(decoded)})
