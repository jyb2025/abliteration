import gc
import torch
import pandas
import argparse
from tqdm import tqdm
from typing import Union
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
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
    choices=["auto", "cuda", "cpu"],
    default="auto",
    help="Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU",
)
parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
parser.add_argument(
    "--skip-begin",
    type=int,
    default=1,
    help="Number of layers to skip at the beginning. Defaults to 1 to avoid messing with the first layer",
)
parser.add_argument(
    "--skip-end", type=int, default=0, help="Number of layers to skip at the end"
)
parser.add_argument(
    "--layer-fraction",
    type=float,
    default=1,
    help="Fraction of layers to use for refusal_dir calculation",
)
parser.add_argument(
    "--scale-factor",
    type=float,
    default=1.0,
    help="Scale factor for ablation. Use a negative scale-factor to encourage refusal. If abliteration is not good, try to increase it a little bit",
)
parser.add_argument(
    "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
)
parser.add_argument(
    "--deccp",
    action="store_true",
    default=False,
    help="For Chinese models, in specific topics",
)
quant = parser.add_mutually_exclusive_group()
quant.add_argument(
    "--load-in-4bit",
    action="store_true",
    default=False,
    help="Load model in 4-bit precision using bitsandbytes",
)
quant.add_argument(
    "--load-in-8bit",
    action="store_true",
    default=False,
    help="Load model in 8-bit precision using bitsandbytes",
)
args = parser.parse_args()


def compute_refusals(
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    layer_fraction: float = 0.6,
) -> torch.Tensor:
    df = pandas.read_parquet("./harmless.parquet")
    harmless_list = df["text"].tolist()
    
    if args.deccp:
        deccp_list = load_dataset("augmxnt/deccp", split="censored")
        harmful_list = deccp_list["text"]
    else:
        df = pandas.read_parquet("./harmful.parquet")
        harmful_list = df["text"].tolist()

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

    torch.cuda.empty_cache()
    gc.collect()

    harmful_outputs = []
    harmless_outputs = []

    for token in tqdm(harmful_tokens, desc="Generating harmful outputs"):
        harmful_outputs.append(
            model.generate(
                token.to("cpu"),
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        )
    for token in tqdm(harmless_tokens, desc="Generating harmless outputs"):
        harmless_outputs.append(
            model.generate(
                token.to("cpu"),
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        )

    torch.cuda.empty_cache()
    gc.collect()

    layer_idx = int(len(model.model.layers) * layer_fraction)
    pos = -1
    harmful_hidden = [
        output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs
    ]
    harmless_hidden = [
        output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs
    ]

    harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
    harmless_mean = torch.stack(harmless_hidden).mean(dim=0)
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    print(refusal_dir)
    return refusal_dir


def modify_tensor(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0
) -> torch.nn.Parameter:
    if tensor_data.device != refusal_dir.device:
        refusal_dir = refusal_dir.to(tensor_data.device)
    tensor_float32 = tensor_data.to(torch.float32)
    refusal_dir_float32 = refusal_dir.to(torch.float32)
    # Ensure refusal_dir is a 1-dimensional tensor
    if refusal_dir_float32.dim() > 1:
        refusal_dir_float32 = refusal_dir_float32.view(-1)
    tensor_float32 -= scale_factor * torch.matmul(
        torch.outer(refusal_dir_float32, refusal_dir_float32), tensor_float32
    )
    tensor_modified = tensor_float32.to(torch.float16)

    torch.cuda.empty_cache()
    gc.collect()

    return torch.nn.Parameter(tensor_modified)


def apply_abliteration(
    model: PreTrainedModel,
    refusal_dir: torch.Tensor,
    skip_begin_layers: int = 1,
    skip_end_layers: int = 0,
    scale_factor: float = 1.0,
) -> PreTrainedModel:
    lm_model = model.model
    assert hasattr(
        lm_model, "layers"
    ), "The model does not have the expected structure."
    num_layers = len(lm_model.layers)
    for layer_idx in tqdm(
        range(skip_begin_layers, num_layers - skip_end_layers),
        desc="Applying abliteration",
    ):
        lm_model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor(
            lm_model.layers[layer_idx].self_attn.o_proj.weight.data,
            refusal_dir,
            scale_factor,
        )
        lm_model.layers[layer_idx].mlp.down_proj.weight = modify_tensor(
            lm_model.layers[layer_idx].mlp.down_proj.weight.data,
            refusal_dir,
            scale_factor,
        )

    torch.cuda.empty_cache()
    gc.collect()

    return model


if __name__ == "__main__":
    assert args.skip_begin >= 1, "Do not mess with the first layer!"
    assert (
        args.layer_fraction >= 0.0 and args.layer_fraction <= 1.0
    ), "Invalid layer fraction"
    torch.inference_mode()
    torch.set_grad_enabled(False)
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=args.device,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )
    model.requires_grad_(False)
    
    assert args.skip_begin + args.skip_end < len(
        model.model.layers
    ), "Too many layers to skip"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, device_map=args.device
    )

    print("Computing refusal dir...")
    refusal_dir = compute_refusals(model, tokenizer, args.layer_fraction)
    print("Applying refusal dir...")

    if args.device != "cpu":
        # WARNING: Reloading model to CPU to apply abliteration is necessary, for cuda device will add slight error to other modules such as q,k,v proj or mlp, and ends up messing up the model.
        print("Reloading model to CPU...")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

    model = apply_abliteration(
        model, refusal_dir, args.skip_begin, args.skip_end, args.scale_factor
    )
    print(f"Saving abliterated model to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
